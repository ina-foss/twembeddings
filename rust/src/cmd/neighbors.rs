use std::boxed::Box;
use std::collections::HashMap;
use std::error::Error;

use clap::Clap;
use serde::Deserialize;

use crate::cli_utils::{acquire_progress_indicator, acquire_tokenizer, get_column_index};
use crate::clustering::ClusteringBuilder;
use crate::vectorization::vectorize;

#[derive(Debug, Deserialize)]
struct VocRecord {
    token: String,
    df: usize,
    idf: f64,
}

#[derive(Clap, Debug)]
#[clap(about = "Find tweets nearest neighbors.")]
pub struct Opts {
    voc_input: String,
    input: String,
    #[clap(long, default_value = "5")]
    query_size: u8,
    #[clap(long, default_value = "64")]
    max_candidates_per_dimension: usize,
    #[clap(long)]
    total: Option<u64>,
    #[clap(long, short, default_value = "0.7")]
    threshold: f64,
    #[clap(long, short, default_value = "0.0")]
    idf_threshold: f64,
    #[clap(long)]
    tsv: bool,
    #[clap(short, long)]
    binary: bool,
    #[clap(short, long)]
    window: usize,
}

pub fn run(cli_args: &Opts) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(&cli_args.voc_input)?;

    let mut wtr = csv::Writer::from_writer(std::io::stdout());

    let bar = acquire_progress_indicator("Compiling vocabulary", None);

    // 1. Compiling vocabulary
    let mut vocabulary: HashMap<String, (usize, f64)> = HashMap::new();

    for (i, result) in rdr.deserialize().enumerate() {
        bar.inc(1);
        let record: VocRecord = result?;
        vocabulary.insert(record.token, (i, record.idf));
    }

    bar.finish_at_current_pos();

    eprintln!("Found {:?} distinct tokens.", vocabulary.len());

    // 2. Streaming tweets, vectorizing and clustering
    let bar = acquire_progress_indicator("Processing tweets", cli_args.total);

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(if cli_args.tsv { b'\t' } else { b',' })
        .from_path(&cli_args.input)?;

    let headers = rdr.headers()?;

    let text_column_index = get_column_index(&headers, "text")?;
    let id_column_index = get_column_index(&headers, "id")?;

    let tokenizer = acquire_tokenizer(None)?;
    let mut clustering = ClusteringBuilder::new(vocabulary.len(), cli_args.window)
        .with_threshold(cli_args.threshold)
        .with_query_size(cli_args.query_size)
        .with_max_candidates_per_dimension(cli_args.max_candidates_per_dimension)
        .build();

    write_csv_record!(wtr, ["id", "nearest_neighbor", "thread_id", "distance"]);

    for (i, result) in rdr.records().enumerate() {
        bar.inc(1);

        let record = result?;

        let text_cell = &record[text_column_index];
        let tweet_id: u64 = record[id_column_index].parse()?;

        let tokens = tokenizer.tokenize(text_cell, cli_args.binary);

        let vector = vectorize(&vocabulary, &tokens, cli_args.idf_threshold, cli_args.binary);

        let clustering_result = clustering.nearest_neighbor(i, tweet_id, vector);

        write_csv_record!(
            wtr,
            match clustering_result.0 {
                Some((nn_id, d)) => vec![
                    tweet_id.to_string(),
                    nn_id.to_string(),
                    clustering_result.1.to_string(),
                    d.to_string()
                ],
                None => vec![
                    tweet_id.to_string(),
                    "".to_string(),
                    clustering_result.1.to_string(),
                    "".to_string()
                ],
            }
        );
    }

    bar.finish_at_current_pos();

    Ok(())
}
