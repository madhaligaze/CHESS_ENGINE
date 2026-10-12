//! Magician: a human-like chess engine.
//!
//! The engine is a conventional alpha-beta searcher (for tactical soundness) whose root
//! decision is made together with a neural network trained to predict the moves of strong
//! human players. See README.md for the design.

pub mod bitboard;
pub mod board;
pub mod datagen;
pub mod engine;
pub mod eval;
pub mod features;
pub mod humaneval;
pub mod movegen;
pub mod nn;
pub mod perft;
pub mod pgn;
pub mod san;
pub mod search;
pub mod tt;
pub mod types;
pub mod uci;
pub mod zobrist;
