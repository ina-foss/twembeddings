use std::borrow::Cow;
use std::collections::HashMap;
use std::error::Error;

use csv::Writer;
use indicatif::{ProgressBar, ProgressStyle};

use crate::stop_words::{STOP_WORDS_EN, STOP_WORDS_FR};
use crate::tokenization::Tokenizer;

macro_rules! write_csv_record {
    ($wtr: expr, $items: expr) => {{
        let mut record = csv::StringRecord::new();
        for item in $items.iter() {
            record.push_field(item);
        }
        $wtr.write_record(&record)?;
    }};
}

pub fn get_column_index(headers: &csv::StringRecord, column_name: &str) -> Result<usize, String> {
    headers.iter().position(|v| v == column_name).ok_or(format!(
        "\"{:?}\" column does not exist in given CSV file!",
        column_name
    ))
}

pub struct ReorderedWriter<'a, W: std::io::Write> {
    writer: &'a mut Writer<W>,
    next_index_to_write: usize,
    buffer: HashMap<usize, Vec<String>>,
}

impl<'a, W: std::io::Write> ReorderedWriter<'a, W> {
    pub fn new(writer: &'a mut Writer<W>) -> Self {
        ReorderedWriter {
            writer,
            next_index_to_write: 0,
            buffer: HashMap::new(),
        }
    }

    pub fn write_vec(&mut self, index: usize, row: Vec<String>) {
        self.buffer.insert(index, row);

        while let Some(other_row) = self.buffer.remove(&self.next_index_to_write) {
            self.writer
                .write_record(&csv::StringRecord::from(other_row))
                .expect("error when writing row");

            self.next_index_to_write += 1;
        }
    }
}

pub fn acquire_progress_indicator(
    msg: impl Into<Cow<'static, str>>,
    total: Option<u64>,
) -> ProgressBar {
    let bar = match total {
        Some(total_count) => {
            let bar = ProgressBar::new(total_count);

            bar.set_style(ProgressStyle::default_bar().template(
                "{msg}: [{elapsed_precise}] < [{eta_precise}] {per_sec} {wide_bar} {pos:>7}/{len:7}",
            ));

            bar
        }
        None => {
            let bar = ProgressBar::new_spinner();

            bar.set_style(
                ProgressStyle::default_spinner()
                    .template("{msg}: {spinner} [{elapsed_precise}] {per_sec} {pos}"),
            );

            bar
        }
    };

    bar.set_message(msg);
    bar.set_draw_rate(1);

    bar
}

pub fn acquire_tokenizer(stopwords: Option<&String>) -> Result<Tokenizer, Box<dyn Error>> {
    let mut tokenizer = Tokenizer::new();

    if let Some(stopwords_path) = stopwords {
        let mut rdr = csv::ReaderBuilder::new().from_path(stopwords_path)?;
        for record in rdr.deserialize::<String>() {
            tokenizer.add_stop_word(&record?);
        }
    }

    for stopword in STOP_WORDS_FR.iter() {
        tokenizer.add_stop_word(stopword);
    }

    for stopword in STOP_WORDS_EN.iter() {
        tokenizer.add_stop_word(stopword);
    }

    Ok(tokenizer)
}
