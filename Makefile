install:
	@echo "Installing..."
	pip install -r requirements.txt

exe_preprocessing:
	@echo "Executing Preprocessing Steps..."
	python src/preprocessing.py data/raw/ data/processed

gen_oversampler:
	@echo "Generates And Saving Synthetic Samples - OVERSAMPLER....."
	python src/oversampler.py data/processed data/processed/sample_data

analyze_synthetic_data:
	@echo "Analyzing Synthetic data..."
	python src/Analyzing_Synthetic_Data.py

base_model_evaluation:
	@echo "Base Model Evaluation With Synthetic Data..."
	python src/base_model_evaluation.py

base_model_evaluation_FS:
	@echo "Base Model Evaluation With Synthetic Data - Feature Selection..."
	python src/classification_model_evaluation_FS.py

feature_selection:
	@echo "Feature Selection..."
	python src/feature_selection.py

feature_selection_eval:
	@echo "Evaluation after Feature Selection..."
	python src/feature_selection_analysis.py

hyper_parametr_tuning:
	@echo "Doing Hyper-Parameter Tuning..."
	python src/hp_tunning_LightGBM.py

evaluate_test_tunned_model:
	@echo "Saving and Evaluating tunned ExtraTreee Model..."
	python src/train_and_evaluate_model.py

setup: install exe_preprocessing gen_oversampler analyze_synthetic_data base_model_evaluation base_model_evaluation_FS feature_selection feature_selection_eval hyper_parametr_tuning evaluate_test_tunned_model


exe_dev_preprocessing:
	@echo "Executing Dev App Preprocessing Steps..."
	python src/dev/preprocessing.py data/raw/ data/processed

gen_dev_oversampler:
	@echo "Generates And Saving Synthetic Samples - OVERSAMPLER for DEV....."
	python src/dev/preprocessing_oversampler.py data/processed data/processed/sample_data

exe_dev_train:
	@echo "Executing Dev App Preprocessing Steps..."
	python src/dev/train.py

exe_run_app:
	@echo "Running Streamlit Apllication..."
	streamlit run '.\1_🚀Project - Income Inequality prediction.py'

dep: exe_dev_preprocessing gen_dev_oversampler exe_dev_preprocessing exe_run_app