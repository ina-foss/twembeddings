use std::collections::HashMap;

use csv::Writer;
use indicatif::{ProgressBar, ProgressStyle};

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

pub fn acquire_progress_indicator(total: Option<u64>) -> ProgressBar {
    match total {
        Some(total_count) => {
            let bar = ProgressBar::new(total_count);

            bar.set_style(ProgressStyle::default_bar().template(
                "[{elapsed_precise}] < [{eta_precise}] {per_sec} {bar:70} {pos:>7}/{len:7}",
            ));

            bar
        }
        None => {
            let bar = ProgressBar::new_spinner();

            bar.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner} [{elapsed_precise}] {per_sec} {pos}"),
            );

            bar
        }
    }
}
