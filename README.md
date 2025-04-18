# Alignment Arena

## Overview

Alignment Arena is a research project that measures bias in masked language models across multiple demographic dimensions including gender, race, religion, culture, and ethnicity. The project aims to demonstrate how model biases impact production safety when these identity categories matter, bringing greater attention to alignment challenges in AI.

By quantifying and highlighting these biases, we reveal how human-created training data can produce models that reflect and sometimes amplify societal prejudices.

## Methodology

Research paper with detailed description will be released on May 2025.

Our bias quantification approach measures how language models reflect societal biases in their predictions. We use various evaluation models to assess bias in filled mask predictions for neutral sentence templates:

- UnBIAS
- d4data
- ModernBERT-bias
- ALBERT-bias

The methodology calculates normalized bias scores across different demographic categories and provides both a bias percentage and an average bias score for each model evaluated.

## Key Features

- Comprehensive evaluation of language model biases across multiple identity categories
- Normalized bias scoring system that works across different evaluation models
- Visual representation of bias metrics through an interactive web interface
- Detailed analysis of how different demographic groups are represented in model outputs

## Data Structure

The bias evaluation data is stored in JSON format containing bias measurements for various language models across four main categories:

- Race/Ethnicity
- Sex/Gender
- Culture/Nationality
- Religion

## Contact

- Email: hr328@cornell.edu
- Phone: +1 (607) 663-1415
- Address: Ithaca, NY
  
## License

This project is licensed under the MIT License - see the LICENSE file for details.
