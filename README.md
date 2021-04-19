- Our solution of idao-2021 consists of the following steps:

1) Define education_bool feature that checks if the education column in client.csv is equal to NaN or PRIMARY_PROFESSIONAL; If it's true, we mark these samples with sale_flg = 0;
2) train lightgbm (model description can be found in submission/SimpleModel.py) on funnel.csv for the rest of the samples. 

- The most simplistic approach without the second step (marking  sale_flg = 0 based on education_bool, and sale_flg = 1 for all other samples) scored ~5400 on the public dataset. We managed to improve that solution by lightgbm only by 60 points on public.

- Check functions/aux_functions.py, submission/train_model.py, submission/generate_submission.py and submission/SimpleModel.py for more details.

- For training run: docker-compose -f docker-compose.train.yaml up

- For validation run: docker-compose -f docker-compose.test.yaml up 

- Our final submission can be found in the folder final_submission.
