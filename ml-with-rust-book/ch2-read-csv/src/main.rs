use std::fs::File;

use csv::Reader;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Record {
    #[serde(rename = "Age")]
    age: u32,
    #[serde(rename = "Sex")]
    gender: String,
    #[serde(rename = "Heart Rate")]
    heart_rate: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open the file for the first pass
    let file = File::open("../data/heart_attack_prediction_dataset.csv")?;
    let mut reader = Reader::from_reader(file);
    read_headers_and_rows(&mut reader)?;

    // Open the file again for the second pass
    let file = File::open("../data/heart_attack_prediction_dataset.csv")?;
    let mut reader = Reader::from_reader(file);
    read_as_records(&mut reader)?;

    Ok(())
}

fn read_headers_and_rows(reader: &mut Reader<File>) -> Result<(), Box<dyn std::error::Error>> {
    let headers = reader.headers()?.clone();
    println!("Headers: {:?}", headers);

    let header_col = headers
        .iter()
        .enumerate()
        .find_map(|(i, h)| if h == "Heart Rate" { Some(i) } else { None })
        .unwrap();

    for result in reader.records() {
        let record = result?;
        let heart_rate = record[header_col].parse::<u32>()?;
        println!("Heart Rate: {}", heart_rate);
    }

    Ok(())
}

fn read_as_records(reader: &mut Reader<File>) -> Result<(), Box<dyn std::error::Error>> {
    let mut records: Vec<Record> = Vec::new();

    for result in reader.deserialize() {
        let record: Record = result?;
        records.push(record);
    }

    dbg!(&records[..5]); // Print first 5 records for verification

    Ok(())
}
