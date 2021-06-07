use std::borrow::Cow;
use std::collections::HashMap;

use csv::Writer;
use indicatif::{ProgressBar, ProgressStyle};

use crate::stop_words::{STOP_WORDS_EN, STOP_WORDS_FR};
use crate::tokenization::Tokenizer;

pub struct ReorderedWriter<'a, W: std::io::Write> {
    writer: &'a mut Writer<W>,
    next_index_to_write: usize,
    buffer: HashMap<usize, Vec<String>>,
}

impl<'a, W: std::io::Write> ReorderedWriter<'a, W> {
    pub fn new(writer: &'a mut Writer<W>) -> ReorderedWriter<'a, W> {
        ReorderedWriter {
            writer,
            next_index_to_write: 0,
            buffer: HashMap::new(),
        }
    }

    pub fn write_vec(&mut self, index: usize, row: Vec<String>) {
        self.buffer.insert(index, row);

        loop {
            match self.buffer.remove(&self.next_index_to_write) {
                Some(other_row) => {
                    self.writer
                        .write_record(&csv::StringRecord::from(other_row))
                        .expect("error when writing row");

                    self.next_index_to_write += 1;
                }
                None => break,
            }
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

    bar
}

pub fn acquire_tokenizer() -> Tokenizer {
    let mut tokenizer = Tokenizer::new();

    for stopword in STOP_WORDS_FR.iter() {
        tokenizer.add_stop_word(stopword);
    }

    for stopword in STOP_WORDS_EN.iter() {
        tokenizer.add_stop_word(stopword);
    }

    tokenizer
}
