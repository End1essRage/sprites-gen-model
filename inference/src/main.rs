use anyhow::{anyhow, Result};
use image::{ImageBuffer, Rgba};
use ndarray::{Array2, s};
use ort::{
    ep,
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    // ==========================================
    // 0. Настройки
    // ==========================================
    // В реальном приложении это будет приходить из запроса
    let prompt_text = "a red sword"; 
    let canvas_size = 16; // 16x16 пикселей
    
    // Простейшая палитра для теста (16 цветов + прозрачность на индексе 0)
    // В реальном проекте ты будешь брать её из конфига или БД
    let palette: Vec<[u8; 4]> = vec![
        [0, 0, 0, 0],       // 0: Прозрачный
        [255, 0, 0, 255],   // 1: Красный
        [0, 255, 0, 255],   // 2: Зелёный
        [0, 0, 255, 255],   // 3: Синий
        [255, 255, 0, 255], // 4: Жёлтый
        [255, 0, 255, 255], // 5: Пурпурный
        [0, 255, 255, 255], // 6: Голубой
        [255, 255, 255, 255],// 7: Белый
        [128, 128, 128, 255],// 8: Серый
        [0, 0, 0, 255],     // 9: Чёрный
        [128, 0, 0, 255],   // 10
        [0, 128, 0, 255],   // 11
        [0, 0, 128, 255],   // 12
        [128, 128, 0, 255], // 13
        [128, 0, 128, 255], // 14
        [0, 128, 128, 255], // 15
    ];

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
        ep::CUDA::default().build(), // <-- Используем GPU
        ep::CPU::default().build(),  // <-- Фолбэк на CPU (если GPU не справится)
    ])
    .map_err(|e| anyhow!("Execution provider error: {e}"))?
    .commit_from_file("minilm.onnx")
    .map_err(|e| anyhow!("Commit error: {e}"))?;

    let mut pixel_model = Session::builder(&environment)
    .map_err(|e| anyhow!("Session builder error: {e}"))?
    .with_optimization_level(GraphOptimizationLevel::Level3)
    .map_err(|e| anyhow!("Optimization level error: {e}"))?
    .with_execution_providers([
        ep::CUDA::default().build(), // <-- Используем GPU
        ep::CPU::default().build(),  // <-- Фолбэк на CPU
    ])
    .map_err(|e| anyhow!("Execution provider error: {e}"))?
    .commit_from_file("pixel_model.onnx")
    .map_err(|e| anyhow!("Commit error: {e}"))?;

    // Загружаем токенизатор
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
    // Создаем тензор для token_type_ids
    let type_ids_tensor = ort::value::Tensor::from_array(
        ([1usize, seq_len as usize], token_type_ids)
    )?;

    
    // Создаем тензоры через Tensor::from_array с кортежем (shape, data)
    // Это работает для i64, в отличие от TensorRef::from_array_view
    let input_ids_tensor = ort::value::Tensor::from_array(
        ([1usize, seq_len as usize], input_ids)
    )?;
    let mask_tensor = ort::value::Tensor::from_array(
        ([1usize, seq_len as usize], attention_mask)
    )?;

    // Передаем по значению (без &)
    let minilm_outputs = minilm.run(inputs![
        "input_ids" => input_ids_tensor,
        "attention_mask" => mask_tensor,
        "token_type_ids" => type_ids_tensor,
    ])?;

    // Достаем last_hidden_state: [1, seq_len, 384]
    let last_hidden_state = minilm_outputs[0]
        .try_extract_array::<f32>()?
        .into_dimensionality::<ndarray::Ix3>()?;

    // Mean Pooling: усредняем по оси токенов, получаем [1, 384]
    let text_embedding = last_hidden_state.mean_axis(ndarray::Axis(1)).unwrap();
    println!("Текстовый эмбеддинг получен. Форма: {:?}", text_embedding.shape());

    // ==========================================
    // 3. Авторегрессионная генерация пикселей
    // ==========================================
    println!("Начинаем генерацию {}x{} пикселей...", canvas_size, canvas_size);
    
    let total_pixels = canvas_size * canvas_size;
    let mut generated_tokens: Vec<i64> = vec![0]; // Стартовый токен (прозрачный)

    // Готовим тензор эмбеддинга один раз (он не меняется)

    for i in 0..total_pixels {
        let current_seq_len = generated_tokens.len();
        
        // Модель требует минимум context_size (16) пикселей на входе
        // Если сгенерировано меньше, добиваем нулями (прозрачными)
        let mut input_vec = generated_tokens.clone();
        if input_vec.len() < 16 {
            input_vec.resize(16, 0);
        }

        let input_tensor = ort::value::Tensor::from_array(
            ([1usize, input_vec.len()], input_vec)
        )?;

        let text_emb_tensor = ort::value::Tensor::from_array(text_embedding.clone())?;

        let outputs = pixel_model.run(inputs![
            "pixel_tokens" => input_tensor,
            "text_embedding" => text_emb_tensor,
        ])?;

        // Модель возвращает [1, vocab_size]. Берем [0, :]
        let logits = outputs["logits"].try_extract_array::<f32>()?;
        let last_logits = logits.slice(s![0, ..]); // <-- ТЕПЕРЬ ТАК
        
        // Находим индекс максимального значения (argmax)
        let mut max_idx = 0;
        let mut max_val = f32::MIN;
        for (idx, &val) in last_logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }

        generated_tokens.push(max_idx as i64);
        
        if (i + 1) % 32 == 0 {
            print!(".");
            use std::io::Write;
            std::io::stdout().flush()?;
        }
    }
    println!("\nГенерация завершена.");

    // ==========================================
    // 4. Сборка картинки из токенов
    // ==========================================
    println!("Сборка изображения...");
    
    // Убираем первый стартовый токен
    let pixel_tokens = &generated_tokens[1..];
    
    // Создаем буфер изображения 16x16
    let mut img = ImageBuffer::new(canvas_size as u32, canvas_size as u32);

    for (i, &token) in pixel_tokens.iter().enumerate() {
        let x = (i % canvas_size) as u32;
        let y = (i / canvas_size) as u32;
        
        // Берем цвет из палитры по индексу токена
        // Если токен выходит за пределы палитры, ставим прозрачный
        let color = palette.get(token as usize).copied().unwrap_or([0, 0, 0, 0]);
        
        img.put_pixel(x, y, Rgba(color));
    }

    // Сохраняем результат
    img.save("output.png")?;
    println!("Готово! Изображение сохранено в output.png");

    Ok(())
}