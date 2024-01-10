# Content-Based Filtering for Predicting Mitre ATT&CK Techniques using Deep Learning
# Project structure
- `configs` : Config files (models and training hyperparameters)  
- `data`:
    - `external`: Data from third party sources.  
    - `interim`: Intermediate data that has been transformed.  
    - `processed`: The final, canonical data sets for modeling.  
    - `raw`: The original, immutable data dump.  
    - `test`: Data for evaluation
    - `lookup_tables`: store look-up tables created after training the models
- `docs`: project documentation
- `src`: source code used in this project
    - `data`: source code for data retrieval and engineering
    - `models`: source code for models
    - `vizualization`: source code for vizualization
- `scripts`: scripts for data processing, training models, and evaluations
- `models`: trained and serialized models