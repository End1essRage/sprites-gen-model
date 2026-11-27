use anyhow::{anyhow, Result};
use image::{ImageBuffer, Rgba};
use ort::{
    ep,
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    // ==========================================
    // 0. Настройки
    // ==========================================
    let prompt_text = "healing potion"; 
    let canvas_size = 16; // 16x16 пикселей

    println!("Загрузка моделей...");

    // ==========================================
    // 1. Загрузка моделей и токенизатора
    // ==========================================
    let environment = ort::init().build()?;

    let mut minilm = Session::builder(&environment)
        .map_err(|e| anyhow!("Session builder error: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("Optimization level error: {e}"))?
        .with_execution_providers([
            ep::CUDA::default().build(), // Если нет видеокарты Nvidia, замени на CPU
            ep::CPU::default().build(),
        ])
        .map_err(|e| anyhow!("Execution provider error: {e}"))?
        .commit_from_file("minilm.onnx")
        .map_err(|e| anyhow!("Commit error: {e}"))?;

    let mut pixel_model = Session::builder(&environment)
        .map_err(|e| anyhow!("Session builder error: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("Optimization level error: {e}"))?
        .with_execution_providers([
            ep::CUDA::default().build(),
            ep::CPU::default().build(),
        ])
        .map_err(|e| anyhow!("Execution provider error: {e}"))?
        .commit_from_file("pixel_model.onnx")
        .map_err(|e| anyhow!("Commit error: {e}"))?;

    let tokenizer = Tokenizer::from_file("tokenizer.json").map_err(|e| anyhow!("Tokenizer: {e}"))?;

    // ==========================================
    // 2. Текст -> Токены -> MiniLM -> Вектор
    // ==========================================
    println!("Кодирование текста: '{}'", prompt_text);
    
    let encoding = tokenizer.encode(prompt_text, true).map_err(|e| anyhow!("Encode: {e}"))?;
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
    let seq_len = input_ids.len() as i64;
    let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&id| id as i64).collect();

    let type_ids_tensor = Tensor::from_array(
        ([1usize, seq_len as usize], token_type_ids)
    )?;
    let input_ids_tensor = Tensor::from_array(
        ([1usize, seq_len as usize], input_ids)
    )?;
    let mask_tensor = Tensor::from_array(
        ([1usize, seq_len as usize], attention_mask)
    )?;

    let minilm_outputs = minilm.run(inputs![
        "input_ids" => input_ids_tensor,
        "attention_mask" => mask_tensor,
        "token_type_ids" => type_ids_tensor,
    ])?;

    let last_hidden_state = minilm_outputs[0]
        .try_extract_array::<f32>()?
        .into_dimensionality::<ndarray::Ix3>()?;

    let text_embedding = last_hidden_state.mean_axis(ndarray::Axis(1)).unwrap();
    println!("Текстовый эмбеддинг получен. Форма: {:?}", text_embedding.shape());

    // ==========================================
    // 3. Генерация изображения
    // ==========================================
    println!("Генерация {}x{} пикселей...", canvas_size, canvas_size);
    
    let text_emb_tensor = Tensor::from_array(text_embedding.clone())?;

    // ОДИН ВЫЗОВ МОДЕЛИ (без палитры!)
    let outputs = pixel_model.run(inputs![
        "text_embedding" => text_emb_tensor,
    ])?;

    // Достаем цвета: [1, 256, 3]
    let colors = outputs["pixel_colors"].try_extract_array::<f32>()?;
    
    println!("Генерация завершена.");

    // ==========================================
    // 4. Сборка картинки
    // ==========================================
    println!("Сборка изображения...");
    
    let mut img = ImageBuffer::new(canvas_size as u32, canvas_size as u32);

    for i in 0..256 {
        let x = (i % canvas_size) as u32;
        let y = (i / canvas_size) as u32;
        
        // Достаем RGB для текущего пикселя и переводим из [0.0, 1.0] в [0, 255]
        let r = (colors[[0, i, 0]] * 255.0).clamp(0.0, 255.0) as u8;
        let g = (colors[[0, i, 1]] * 255.0).clamp(0.0, 255.0) as u8;
        let b = (colors[[0, i, 2]] * 255.0).clamp(0.0, 255.0) as u8;
        
        // Прозрачность ставим 255 (непрозрачный). 
        // Если фон должен быть прозрачным, можно вычислять его по условию (например, если r,g,b == 0, то alpha = 0)
        let mut alpha = 255;
        if r == 0 && g == 0 && b == 0 {
            alpha = 0; // Делаем черный цвет прозрачным
        }

        img.put_pixel(x, y, Rgba([r, g, b, alpha]));
    }

    img.save("output.png")?;
    println!("Готово! Изображение сохранено в output.png");

    Ok(())
}