# Transaction Categorization using Machine Learning

## Overview

This project focuses on extracting structured, meaningful information from bank transaction descriptions, which are typically messy, and ambiguous. The goal is to build a machine-learning-driven system that can:

Identify transaction types (e.g., UPI, ATM Withdrawal, ACH Debit, Card Payment).

Extract merchant names from noisy free-text descriptions.

Classify transactions into spending categories (e.g., Food, Utilities, Shopping, Travel).

Prepare high-quality training data for downstream models.

The project is inspired by concepts learned during exploration of the Auth’n’Capture book, which explains the mechanics of the payment ecosystem—debit card number systems, payment gateways, merchant acquiring systems, and settlement logic. This sparked an interest in leveraging that understanding to solve a practical FinTech problem: making sense of transaction logs using NLP.
