# py-hands-ml-tf

The script in main.py recognizes gestures from the dataset https://www.kaggle.com/datasets/datamunge/sign-language-mnist

process_input.py processes csv to produce training, validation and test dataset images.

In the "job_script" directory, the script has been adapted and integrated to run in the Cloud, using AzureML.

You have to add your own json configuration file (which is taken from AzureML web console) to configure the script to run with your own subscription.