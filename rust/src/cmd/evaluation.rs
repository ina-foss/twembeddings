use chrono_tz::Tz;
use clap::Clap;
use compound_duration::format_dhms;
use std::boxed::Box;
use std::collections::HashMap;
use std::error::Error;

use crate::cli_utils::{acquire_progress_indicator, get_column_index};
use crate::date_utils::inferred_date;

#[derive(Clap, Debug)]
#[clap(about = "Evaluate a clustering result.")]
pub struct Opts {
    truth: String,
    predicted: String,
    #[clap(long)]
    total: Option<u64>,
    #[clap(long)]
    tsv: bool,
    #[clap(long, default_value = "created_at")]
    datecol: String,
    ///If timestamps are provided, they will be parsed as UTC timestamps, then converted to
    ///the provided timezone. Other date formats will not be converted.
    #[clap(long, default_value = "Europe/Paris")]
    timezone: String,
}

pub fn run(cli_args: &Opts) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(if cli_args.tsv { b'\t' } else { b',' })
        .from_path(&cli_args.truth)?;

    let bar = acquire_progress_indicator("Indexing truth", None);

    let headers = rdr.headers()?;

    let id_column_index = get_column_index(&headers, "id")?;
    let label_column_index = get_column_index(&headers, "label")?;

    let mut truth: HashMap<u64, usize> = HashMap::new();
    let mut truth_clusters: HashMap<usize, Vec<u64>> = HashMap::new();

    for result in rdr.records() {
        bar.inc(1);

        let record = result?;
        let id: u64 = record[id_column_index].parse()?;
        let label_option: Option<usize> = record[label_column_index]
            .split('.')
            .next()
            .unwrap()
            .parse()
            .ok();

        if let Some(label) = label_option {
            truth.insert(id, label);
            let cluster = truth_clusters.entry(label).or_default();
            cluster.push(id);
        }
    }

    bar.finish_at_current_pos();

    let mut rdr = csv::ReaderBuilder::new().from_path(&cli_args.predicted)?;

    let headers = rdr.headers()?;

    let id_column_index = get_column_index(&headers, "id")?;
    let thread_column_index = get_column_index(&headers, "thread_id")?;
    let date_column_index = get_column_index(&headers, &cli_args.datecol)?;

    let bar = acquire_progress_indicator("Processing predictions", cli_args.total);

    let mut predicted: HashMap<u64, usize> = HashMap::new();
    let mut predicted_clusters_dates: HashMap<usize, (usize, usize, String, String)> =
        HashMap::new();
    let mut counter = 0;

    let mut start_date: String = String::new();
    let mut end_date: String = String::new();

    for result in rdr.records() {
        bar.inc(1);

        let record = result?;

        let id: u64 = record[id_column_index].parse()?;
        let thread_id: usize = record[thread_column_index].parse()?;
        let tweet_date: String = record[date_column_index].to_string();
        if counter == 0 {
            start_date = tweet_date.clone();
        }
        counter += 1;
        end_date = tweet_date.clone();

        if truth.contains_key(&id) {
            predicted_clusters_dates
                .entry(thread_id)
                .and_modify(|e| *e = (e.0 + 1, e.1 + 1, e.2.clone(), tweet_date.clone()))
                .or_insert((1, 1, tweet_date.clone(), tweet_date.clone()));
            predicted.insert(id, thread_id);
        } else {
            predicted_clusters_dates
                .entry(thread_id)
                .and_modify(|e| *e = (e.0, e.1 + 1, e.2.clone(), tweet_date.clone()))
                .or_insert((0, 1, tweet_date.clone(), tweet_date.clone()));
        }
    }

    bar.finish_at_current_pos();

    // Producing statistics about the length of events
    let tz: Tz = cli_args.timezone.parse().or(Err("Unknown timezone"))?;
    let start_day = inferred_date(&start_date, &tz)?
        .format("%Y-%m-%d")
        .to_string();
    let end_day = inferred_date(&end_date, &tz)?
        .format("%Y-%m-%d")
        .to_string();

    let bar = acquire_progress_indicator(
        "Running descriptive statistics",
        Some(predicted_clusters_dates.len() as u64),
    );
    let mut sum_duration = 0;
    let mut events_starting_on_start_date_count = 0;
    let mut events_ending_on_end_date_count = 0;
    let mut events_covering_whole_period_count = 0;
    let mut nb_clusters = 0;

    for (_nb_annotated_tweeets, nb_tweets, first_tweet_date, last_tweet_date) in
        predicted_clusters_dates.values()
    {
        bar.inc(1);
        if nb_tweets <= &1 {
            continue;
        }
        nb_clusters += 1;
        let inferred_last_tweet_day = inferred_date(&last_tweet_date, &tz)?
            .format("%Y-%m-%d")
            .to_string();
        if inferred_date(&first_tweet_date, &tz)?
            .format("%Y-%m-%d")
            .to_string()
            == start_day
        {
            events_starting_on_start_date_count += 1;
            if inferred_last_tweet_day == end_day {
                events_covering_whole_period_count += 1;
            }
        }
        if inferred_date(&last_tweet_date, &tz)?
            .format("%Y-%m-%d")
            .to_string()
            == end_day
        {
            events_ending_on_end_date_count += 1;
        }

        let first_datetime =
            inferred_date(&first_tweet_date, &tz).or(Err("Unknown date format!"))?;
        let last_datetime = inferred_date(&last_tweet_date, &tz).or(Err("Unknown date format!"))?;
        let duration = last_datetime
            .signed_duration_since(first_datetime)
            .num_seconds();
        sum_duration += duration;
    }

    bar.finish_at_current_pos();

    // Running the actual evaluation using best matching scheme
    let bar = acquire_progress_indicator("Running evaluation", Some(truth_clusters.len() as u64));

    let mut precision = 0.0;
    let mut recall = 0.0;
    let mut f1 = 0.0;
    let mut n: usize = 0;

    for truth_cluster in truth_clusters.values() {
        bar.inc(1);

        let mut candidates: HashMap<usize, usize> = HashMap::new();
        let mut truth_cluster_size: usize = 0;

        for truth_id in truth_cluster {
            match predicted.get(truth_id) {
                Some(&candidate_thread_id) => {
                    candidates
                        .entry(candidate_thread_id)
                        .and_modify(|x| *x += 1)
                        .or_insert(1);

                    truth_cluster_size += 1;
                }
                None => {
                    continue;
                }
            }
        }

        if truth_cluster_size == 0 {
            continue;
        }

        // Only adding to n if cluster contains any valid tweet
        n += 1;

        let best = candidates
            .iter()
            .map(|(thread_id, true_positives)| {
                let matching_cluster_size = predicted_clusters_dates[thread_id].0;

                let false_positives = (matching_cluster_size - true_positives) as f64;
                let false_negatives = (truth_cluster_size - true_positives) as f64;

                let true_positives = *true_positives as f64;

                let local_precision = true_positives / (true_positives + false_positives);
                let local_recall = true_positives / (true_positives + false_negatives);
                let local_f1 =
                    2.0 * local_precision * local_recall / (local_precision + local_recall);

                (local_precision, local_recall, local_f1)
            })
            .max_by(|x, y| x.2.partial_cmp(&y.2).unwrap())
            .unwrap();

        precision += best.0;
        recall += best.1;
        f1 += best.2;
    }

    let n = n as f64;

    precision /= n;
    recall /= n;
    f1 /= n;

    bar.finish_at_current_pos();
    // precision,recall,f1,nb_tweets,nb_events,nb_events_bigger_than_1,mean_duration,nb_events_first_day,%_events_first_day,nb_events_last_day,%_events_last_day,nb_events_whole,%_events_whole
    let mut wtr = csv::Writer::from_writer(std::io::stdout());
    write_csv_record!(
        wtr,
        [
            format!("{:.3}", precision),
            format!("{:.3}", recall),
            format!("{:.3}", f1),
            counter.to_string(),
            predicted_clusters_dates.len().to_string(),
            nb_clusters.to_string(),
            format_dhms((sum_duration as f64 / nb_clusters as f64) as usize),
            events_starting_on_start_date_count.to_string(),
            format!(
                "{:.3}",
                (events_starting_on_start_date_count as f64) / (nb_clusters as f64)
            ),
            events_ending_on_end_date_count.to_string(),
            format!(
                "{:.3}",
                (events_ending_on_end_date_count as f64) / (nb_clusters as f64)
            ),
            events_covering_whole_period_count.to_string(),
            format!(
                "{:.3}",
                (events_covering_whole_period_count as f64) / (nb_clusters as f64)
            ),
        ]
    );

    Ok(())
}
