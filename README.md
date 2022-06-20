# ABCinML

A repository contains the experiment code for the paper _ABCinML: Anticipatory Bias Correction in Machine Learning Applications_ by Abdulaziz A. Almuzaini, Chidansh A. Bhatt, David M. Pennock and Vivek K. Singh.



# Abstract  
The idealization of a static machine-learned model, trained once and deployed forever, is not practical. As input distributions change over time, the model will not only lose accuracy, any constraints to reduce bias against a protected class may fail to work as intended. Thus, researchers have begun to explore ways to maintain algorithmic fairness over time. One line of work focuses on dynamic learning: retraining after each batch, and the other on robust learning which tries to make algorithms robust against all possible future changes. Dynamic learning seeks to reduce biases soon after they have occurred and robust learning often yields (overly) conservative models. We propose an anticipatory dynamic learning approach for correcting the algorithm to mitigate bias before it occurs. Specifically, we make use of anticipations regarding the relative distributions of population subgroups (e.g., relative ratios of male and female applicants) in the next cycle to identify the right parameters for an importance weighing fairness approach. Results from experiments over multiple real-world datasets suggest that this approach has promise for anticipatory bias correction.


# Requirements
 - Download the datasets https://drive.google.com/file/d/1v8Utgi7DbtPLQyh_9Z4taGsrV2qAGX_1/view?usp=sharing
 - Unzip the folder "data.zip"
 - Install packages:
    ```
    pip install -r requirements.txt
   ```
 - Experiment with the main notebook "Final_Model.ipynb"
