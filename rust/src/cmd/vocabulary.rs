use std::boxed::Box;
use std::collections::HashMap;
use std::error::Error;
use std::sync::Mutex;

use clap::Clap;
use rayon::prelude::*;
use serde::Deserialize;

use crate::cli_utils::{acquire_progress_indicator, acquire_tokenizer, get_column_index};

#[derive(Debug, Deserialize)]
struct ExtraneousVocRecord {
    token: String,
    df: usize,
}

struct DocumentFrequencies {
    counter: HashMap<String, usize>,
    min_df: usize,
    total: usize,
}

impl DocumentFrequencies {
    pub fn new(min_df: usize) -> DocumentFrequencies {
        DocumentFrequencies {
            counter: HashMap::new(),
            min_df,
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

    pub fn add_extraneous_doc_count(&mut self, count: usize) {
        self.total += count;
    }

    pub fn add_extraneous_token(&mut self, token: String, df: usize) {
        self.counter
            .entry(token)
            .and_modify(|x| *x += df)
            .or_insert(df);
    }

    pub fn sorted_vocab(&self) -> Vec<(&String, usize, f64)> {
        let mut dfs = self
            .counter
            .iter()
            .filter(|(_, &count)| count >= self.min_df)
            .map(|(token, count)| (*count, token))
            .collect::<Vec<(usize, &String)>>();

        dfs.sort_unstable_by(|a, b| b.cmp(a));

        dfs.iter()
            .map(|(count, token)| {
                let idf = 1.0 + ((self.total as f64 + 1.0) / (*count as f64 + 1.0)).ln();

                (*token, *count, idf)
            })
            .collect()
    }
}

#[derive(Clap, Debug)]
#[clap(about = "Extract tweet vocabulary from the given CSV file.")]
pub struct Opts {
    input: String,
    #[clap(long)]
    merge: Option<String>,
    /// Path (optional) to a custom stopwords list in csv format (one word per row,
    /// with headers). Your stopwords will be added to the default stopwords list.
    #[clap(long)]
    stopwords: Option<String>,
    // NOTE: it is written as 10 in the original implementation but the condition
    // used with it makes it actually 11 conceptually.
    #[clap(long, default_value = "11")]
    min_df: usize,
    #[clap(long)]
    total: Option<u64>,
    #[clap(long)]
    tsv: bool,
}

pub fn run(cli_args: &Opts) -> Result<(), Box<dyn Error>> {
    let mut document_frequencies = DocumentFrequencies::new(cli_args.min_df);

    if let Some(merge_path) = &cli_args.merge {
        let mut rdr = csv::ReaderBuilder::new().from_path(merge_path)?;
        let bar = acquire_progress_indicator("Indexing extraneous vocabulary", None);

        for result in rdr.deserialize() {
            bar.inc(1);

            let record: ExtraneousVocRecord = result?;

            if record.token == "$" {
                document_frequencies.add_extraneous_doc_count(record.df);
            } else {
                document_frequencies.add_extraneous_token(record.token, record.df);
            }
        }

        bar.finish_at_current_pos();
    }

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(if cli_args.tsv { b'\t' } else { b',' })
        .from_path(&cli_args.input)?;

    let mut wtr = csv::Writer::from_writer(std::io::stdout());
    write_csv_record!(wtr, ["token", "df", "idf"]);

    let bar = acquire_progress_indicator("Tokenizing tweets", cli_args.total);

    let headers = rdr.headers()?;

    let text_column_index = get_column_index(&headers, "text")?;

    let tokenizer = acquire_tokenizer(cli_args.stopwords.as_ref())?;

    let mutex = Mutex::new(&mut document_frequencies);

    rdr.records()
        .par_bridge()
        .map(|result| {
            let record = result.expect("Could not read row!");

            tokenizer.tokenize(
                &record
                    .get(text_column_index)
                    .expect("Found a row with fewer columns than expected!"),
                true
            )
        })
        .for_each(|tokens| {
            bar.inc(1);

            let mut document_frequencies = mutex.lock().unwrap();
            document_frequencies.add_doc(tokens);
        });

    bar.finish_at_current_pos();

    let doc_count = document_frequencies.doc_count();
    let voc_size = document_frequencies.voc_size();

    let mut actual_voc_size = 0;

    for (token, df, idf) in document_frequencies.sorted_vocab() {
        write_csv_record!(wtr, [token, &df.to_string(), &idf.to_string()]);
        actual_voc_size += 1;
    }

    wtr.flush()?;

    eprintln!("Tokenized tweets: {:?}", doc_count);
    eprintln!("Total vocab size: {:?}", voc_size);
    eprintln!("Actual vocab size after df trimming: {:?}", actual_voc_size);

    Ok(())
}
