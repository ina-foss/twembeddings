use std::borrow::Cow;
use std::collections::{HashSet, VecDeque};
use std::str::CharIndices;

use itertools::Itertools;
use regex::Regex;
use unidecode::unidecode;

lazy_static! {
    static ref URL_RE: Regex = Regex::new(r"https?://\S+").unwrap();
    static ref MENTION_RE: Regex = Regex::new(r"@\S+").unwrap();
    static ref WORD_RE: Regex = Regex::new(r"#\w+\b|\b\w\w+\b").unwrap();
}

fn reduce_lengthening(string: &str) -> String {
    let mut output: String = String::new();

    let mut counter = 0;
    let mut last_char: Option<char> = None;

    for c in string.chars() {
        match last_char {
            Some(last) => {
                if c == last && !c.is_numeric() {
                    counter += 1;
                } else {
                    counter = 0;
                    last_char = Some(c);
                }
            }
            None => {
                last_char = Some(c);
            }
        }

        if counter < 3 {
            output.push(c);
        }
    }

    output
}

fn strip_urls(string: &str) -> Cow<str> {
    URL_RE.replace_all(string, "")
}

fn strip_mentions(string: &str) -> Cow<str> {
    MENTION_RE.replace_all(string, "")
}

fn is_hashtag(string: &str) -> bool {
    string.len() > 1 && string.starts_with('#')
}

/// Enum representing the hashtag splitter state
enum HashtagSplitterState {
    UpperStart,
    UpperNext,
    Number,
    Lower,
}

use HashtagSplitterState::*;

struct HashtagSplit<'a> {
    text: &'a str,
    offset: usize,
    state: HashtagSplitterState,
    chars: CharIndices<'a>,
    done: bool,
}

impl<'a> HashtagSplit<'a> {
    pub fn new(hashtag: &'a str) -> Self {
        let mut chars = hashtag.char_indices();

        // Consuming the hashtag `#` and first char
        chars.next().unwrap();
        chars.next().unwrap();

        HashtagSplit {
            text: hashtag,
            offset: 1,
            state: UpperStart,
            chars,
            done: false,
        }
    }
}

impl<'a> Iterator for HashtagSplit<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let text = self.text;

        loop {
            match self.chars.next() {
                Some((i, c)) => {
                    let result = match self.state {
                        Lower => {
                            if c.is_uppercase() {
                                (Some(0), UpperStart)
                            } else if c.is_numeric() {
                                (Some(0), Number)
                            } else {
                                (None, Lower)
                            }
                        }
                        UpperStart => {
                            if c.is_lowercase() {
                                (None, Lower)
                            } else if c.is_numeric() {
                                (Some(0), Number)
                            } else {
                                (None, UpperNext)
                            }
                        }
                        UpperNext => {
                            if c.is_lowercase() {
                                (Some(1), Lower)
                            } else if c.is_numeric() {
                                (Some(0), Number)
                            } else {
                                (None, UpperNext)
                            }
                        }
                        Number => {
                            if !c.is_numeric() {
                                if c.is_uppercase() {
                                    (Some(0), UpperStart)
                                } else {
                                    (Some(0), Lower)
                                }
                            } else {
                                (None, Number)
                            }
                        }
                    };

                    self.state = result.1;

                    if let Some(delta) = result.0 {
                        let current_offset = self.offset;
                        self.offset = i - delta;
                        return Some(&text[current_offset..i - delta]);
                    }
                }
                None => {
                    self.done = true;
                    return Some(&text[self.offset..]);
                }
            }
        }
    }
}

fn split_hashtag(hashtag: &str) -> HashtagSplit {
    HashtagSplit::new(hashtag)
}

pub struct Tokens<'a> {
    text: String,
    offset: usize,
    hashtag_split: VecDeque<String>,
    tokenizer: &'a Tokenizer,
}

impl<'a> Tokens<'a> {
    pub fn new(text: &str, tokenizer: &'a Tokenizer) -> Self {
        let text = strip_urls(text);
        let text = if tokenizer.strip_mentions {
            strip_mentions(&text)
        } else {
            text
        };
        let text = unidecode(&text);

        Tokens {
            text,
            offset: 0,
            hashtag_split: VecDeque::new(),
            tokenizer,
        }
    }

    fn post_process(&self, token: &str) -> String {
        reduce_lengthening(&token).to_lowercase()
    }

    fn must_skip_token(&self, token: &str) -> bool {
        if token.len() > 4 && token.chars().all(|c| c.is_numeric()) {
            return true;
        }

        self.tokenizer.stoplist.contains(token)
    }
}

impl<'a> Iterator for Tokens<'a> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let token = if let Some(value) = self.hashtag_split.pop_front() {
                value
            } else {
                match WORD_RE.find_at(&self.text, self.offset) {
                    Some(m) => {
                        self.offset = m.end();

                        let value = m.as_str();

                        if is_hashtag(value) {
                            for word in split_hashtag(value) {
                                self.hashtag_split.push_back(word.to_string());
                            }
                            continue;
                        }

                        value.to_string()
                    }
                    None => return None,
                }
            };

            let token = self.post_process(&token);

            if self.must_skip_token(&token) {
                continue;
            }

            return Some(token);
        }
    }
}

pub struct Tokenizer {
    stoplist: HashSet<String>,
    strip_mentions: bool,
}

impl Default for Tokenizer {
    fn default() -> Tokenizer {
        Tokenizer {
            stoplist: HashSet::new(),
            strip_mentions: false,
        }
    }
}

impl Tokenizer {
    pub fn new() -> Self {
        Tokenizer::default()
    }

    pub fn add_stop_word(&mut self, word: &str) -> &mut Self {
        self.stoplist.insert(word.to_lowercase());
        self
    }

    pub fn tokenize(&self, text: &str, unique: bool) -> Vec<String> {
        let tokens = Tokens::new(text, self);
        if unique {
            return tokens.unique().collect();
        }
        return tokens.collect::<Vec<String>>();
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_reduce_lengthening() {
        assert_eq!(reduce_lengthening("cool"), "cool");
        assert_eq!(reduce_lengthening("coool"), "coool");
        assert_eq!(reduce_lengthening("cooooool"), "coool");
        assert_eq!(reduce_lengthening("cooooooooooolool"), "cooolool");
        assert_eq!(reduce_lengthening("100000000"), "100000000");
    }

    #[test]
    fn test_strip_urls() {
        assert_eq!(
            strip_urls("This is a url: http://lemonde.fr no?"),
            "This is a url:  no?"
        )
    }

    #[test]
    fn test_strip_mentions() {
        assert_eq!(
            strip_mentions("This is a mention: @Yomgui no?"),
            "This is a mention:  no?"
        )
    }

    #[test]
    fn split_hashtag_test() {
        fn collect_split_hashtag(text: &str) -> Vec<&str> {
            split_hashtag(text).collect()
        }

        assert_eq!(collect_split_hashtag("#test"), vec!["test"]);
        assert_eq!(collect_split_hashtag("#Test"), vec!["Test"]);
        assert_eq!(collect_split_hashtag("#t"), vec!["t"]);
        assert_eq!(collect_split_hashtag("#T"), vec!["T"]);
        assert_eq!(
            collect_split_hashtag("#TestWhatever"),
            vec!["Test", "Whatever"]
        );
        assert_eq!(
            collect_split_hashtag("#testWhatever"),
            vec!["test", "Whatever"]
        );
        assert_eq!(
            collect_split_hashtag("#ÉpopéeRusse"),
            vec!["Épopée", "Russe"]
        );
        assert_eq!(
            collect_split_hashtag("#TestOkFinal"),
            vec!["Test", "Ok", "Final"]
        );
        assert_eq!(
            collect_split_hashtag("#TestOkFinalT"),
            vec!["Test", "Ok", "Final", "T"]
        );
        assert_eq!(
            collect_split_hashtag("#Test123Whatever"),
            vec!["Test", "123", "Whatever"]
        );
        assert_eq!(collect_split_hashtag("#TDF2018"), vec!["TDF", "2018"]);
        assert_eq!(collect_split_hashtag("#T2018"), vec!["T", "2018"]);
        assert_eq!(
            collect_split_hashtag("#TheID2018"),
            vec!["The", "ID", "2018"]
        );
        assert_eq!(
            collect_split_hashtag("#8YearsOfOneDirection"),
            vec!["8", "Years", "Of", "One", "Direction"]
        );
        assert_eq!(
            collect_split_hashtag("#This18Gloss"),
            vec!["This", "18", "Gloss"]
        );
        assert_eq!(
            collect_split_hashtag("#WordpressIDInformation"),
            vec!["Wordpress", "ID", "Information"]
        );
        assert_eq!(
            collect_split_hashtag("#LearnWCFInSixEasyMonths"),
            vec!["Learn", "WCF", "In", "Six", "Easy", "Months"]
        );
        assert_eq!(
            collect_split_hashtag("#ThisIsInPascalCase"),
            vec!["This", "Is", "In", "Pascal", "Case"]
        );
        assert_eq!(
            collect_split_hashtag("#whatAboutThis"),
            vec!["what", "About", "This"]
        );
        assert_eq!(
            collect_split_hashtag("#This123thingOverload"),
            vec!["This", "123", "thing", "Overload"]
        );
        assert_eq!(collect_split_hashtag("#final19"), vec!["final", "19"]);
    }

    #[test]
    fn test_tokenize() {
        let default_tokenizer = Tokenizer::new();

        assert_eq!(
        default_tokenizer.tokenize("Hello World, this is I the élémental @Yomgui http://lemonde.fr type looooool! #Whatever", false),
        vec!["hello", "world", "this", "is", "the", "elemental", "yomgui", "type", "loool", "whatever"]
    );

        assert_eq!(
            default_tokenizer.tokenize("Hello #EpopeeRusse! What's brewing?,", false),
            vec!["hello", "epopee", "russe", "what", "brewing"]
        );

        assert_eq!(
            default_tokenizer.tokenize("Hello to this number: 400000 and this one: 34", false),
            vec!["hello", "to", "this", "number", "and", "this", "one", "34"]
        );

        assert_eq!(
            default_tokenizer.tokenize("Hello! hello bonjour hello?", true),
            vec!["hello", "bonjour"]
        );

        let mut tokenizer_with_stopwords = Tokenizer::new();
        tokenizer_with_stopwords.add_stop_word("world");

        assert_eq!(
            tokenizer_with_stopwords.tokenize("Hello World!", false),
            vec!["hello"]
        );
    }
}
