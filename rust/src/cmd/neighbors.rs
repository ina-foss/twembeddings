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
    #[clap(short, long)]
    window: usize,
    #[clap(long)]
    total: Option<u64>,
    #[clap(long)]
    tsv: bool,
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

    let tokenizer = acquire_tokenizer();
    let mut clustering = ClusteringBuilder::new(vocabulary.len(), cli_args.window).build();

    write_csv_record!(wtr, ["id", "nearest_neighbor", "distance"]);

    for (i, result) in rdr.records().enumerate() {
        bar.inc(1);

        let record = result?;

        let text_cell = &record[text_column_index];
        let id_cell = &record[id_column_index];

        let tokens = tokenizer.unique_tokens(text_cell);

        let vector = vectorize(&vocabulary, &tokens);

        let nearest_neighbor = clustering.nearest_neighbor(i, vector);

        write_csv_record!(
            wtr,
            match nearest_neighbor {
                Some((j, d)) => vec![id_cell.into(), j.to_string(), d.to_string()],
                None => vec![id_cell.into(), "".to_string(), "".to_string()],
            }
        );
    }

    bar.finish_at_current_pos();

    Ok(())
}
