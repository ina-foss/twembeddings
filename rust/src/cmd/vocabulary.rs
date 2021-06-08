use std::boxed::Box;
use std::collections::HashMap;
use std::error::Error;
use std::sync::Mutex;

use clap::Clap;
use rayon::prelude::*;

use crate::cli_utils::{acquire_progress_indicator, acquire_tokenizer, get_column_index};

// NOTE: it is written as 10 in the original implementation but the condition
// used with it makes it actually 11 conceptually.
const DF_MIN: usize = 11;

struct DocumentFrequencies {
    counter: HashMap<String, usize>,
    total: usize,
}

impl DocumentFrequencies {
    pub fn new() -> DocumentFrequencies {
        DocumentFrequencies {
            counter: HashMap::new(),
            total: 0,
        }
    }

    pub fn doc_count(&self) -> usize {
        self.total
    }

    pub fn voc_size(&self) -> usize {
        self.counter.len()
    }

    pub fn add_doc(&mut self, tokens: Vec<String>) {
        self.total += 1;

        for token in tokens {
            self.counter
                .entry(token)
                .and_modify(|x| *x += 1)
                .or_insert(1);
        }
    }

    pub fn into_vocab(self) -> impl Iterator<Item = (String, usize, f64)> {
        let mut dfs = self
            .counter
            .into_iter()
            .map(|(token, count)| (count, token))
            .collect::<Vec<(usize, String)>>();

        dfs.sort_unstable_by(|a, b| b.cmp(a));

        let total = self.total;

        dfs.into_iter()
            .filter(|(count, _)| count >= &DF_MIN)
            .map(move |(count, token)| {
                let idf = 1.0 + ((total as f64 + 1.0) / (count as f64 + 1.0)).ln();

                (token, count, idf)
            })
    }
}

#[derive(Clap, Debug)]
#[clap(about = "Extract tweet vocabulary from the given CSV file.")]
pub struct Opts {
    input: String,
    #[clap(long)]
    total: Option<u64>,
    #[clap(long)]
    tsv: bool,
}

pub fn run(cli_args: &Opts) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(if cli_args.tsv { b'\t' } else { b',' })
        .from_path(&cli_args.input)?;

    let mut wtr = csv::Writer::from_writer(std::io::stdout());
    write_csv_record!(wtr, ["token", "df", "idf"]);

    let bar = acquire_progress_indicator("Tokenizing tweets", cli_args.total);

    let headers = rdr.headers()?;

    let text_column_index = get_column_index(&headers, "text")?;

    let tokenizer = acquire_tokenizer();
    let document_frequencies = DocumentFrequencies::new();

    let mutex = Mutex::new(document_frequencies);

    rdr.records()
        .par_bridge()
        .map(|result| {
            let record = result.expect("Could not read row!");

            tokenizer.unique_tokens(
                &record
                    .get(text_column_index)
                    .expect("Found a row with fewer columns than expected!"),
            )
        })
        .for_each(|tokens| {
            bar.inc(1);

            let mut document_frequencies = mutex.lock().unwrap();
            document_frequencies.add_doc(tokens);
        });

    bar.finish_at_current_pos();

    let document_frequencies = mutex.into_inner()?;
    let doc_count = document_frequencies.doc_count();
    let voc_size = document_frequencies.voc_size();

    let mut actual_voc_size = 0;

    for (token, df, idf) in document_frequencies.into_vocab() {
        write_csv_record!(wtr, [token, df.to_string(), idf.to_string()]);
        actual_voc_size += 1;
    }

    wtr.flush()?;

    eprintln!("Tokenized tweets: {:?}", doc_count);
    eprintln!("Total vocab size: {:?}", voc_size);
    eprintln!("Actual vocab size after df trimming: {:?}", actual_voc_size);

    Ok(())
}
