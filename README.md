# ACPLearn: A Deep Neural Network to predict anticancer peptides

Try it out: acp-learn.com

<img width="892" alt="image" src="https://github.com/user-attachments/assets/6e261423-3cc3-4371-bcf9-e3cd82c932b9" />



Left: Breast Cancer Cells

Right: Breast Cancer Cells treated with an anticancer peptide predicted by deep learning model


# Abstract 

Anticancer Peptides (ACPs), a class of small peptide molecules, have gained increasing attention in cancer research due to their ability to selectively target and kill cancer cells, sparing normal cells. ACPs, unlike chemotherapy, have less toxicity and fewer side effects. Unfortunately, the experimental identification of ACPs is time-consuming and expensive, therefore computational methods to identify key features of ACPs is promising. 

Here, a robust deep-learning model was developed that recognized molecular features of ACPs from the CancerPPD database with a sensitivity and specificity of 92% and 99%. Next, 20,000 peptide entries catalogued in the J. Craig Venter Institute PhyloDB database were screened for potentially novel ACPs by the deep learning model, from which the model selected top forty novel ACPs with >90% ACP probability, six of which originated from a species of unicellular algae P. Antarctica. MCF-7 Breast Cancer Cell culture validation of the top four novel ACPs exhibit a statically significant cytotoxic effect on cancer cells at concentrations above 10-100 mcg/ml.

In summary, for the first time, the deep learning approach described here applies learned information on ACPs to make new predictions and would present the first AI predicted novel ACPs to be validated in vitro.

# Methods

For more information about the methodology, visit About.pdf. In summary, I utilized chemical properties of peptides such as AA composition, polarity, etc. for feature engineering to train a DNN. We scrape known ACPs as training data from CancerPPD. ACPLearn beats out other state of the art methods on a validation dataset ACP740


<div align="center">
    <img width="550" alt="image" src="https://github.com/user-attachments/assets/80236740-6431-4e62-92aa-0588e8699797" />
</div>

