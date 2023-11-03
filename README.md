<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
</head>
<body>
  <h1>Project Documentation</h1>
  <p>This project includes scripts for handling imbalanced datasets and applying feature selection methods to improve machine learning models.</p>

  <h2>Script A: Imbalanced Dataset Handling</h2>
  <p><code>scripta.py</code> applies oversampling to balance the class distribution in a dataset.</p>

  <h3>Dependencies</h3>
  <p>The script requires the following Python libraries:</p>
  <ul>
      <li>pandas</li>
      <li>numpy</li>
      <li>imbalanced-learn (imblearn)</li>
  </ul>

  <h3>Usage</h3>
  <p>The dataset should be in a CSV file named <code>dataset.csv</code> in the same directory as the script. The script can be run with the following command:</p>
  <pre><code>python scripta.py</code></pre>

  <h3>Functionality</h3>
  <p>The script performs the following steps:</p>
  <ol>
      <li>Reads the dataset and separates features from the target variable.</li>
      <li>Calculates the initial class distribution.</li>
      <li>Applies RandomOverSampler to balance the classes.</li>
      <li>Calculates and displays the new class distribution.</li>
  </ol>

  <h3>Expected Outputs</h3>
  <p>The script will output the class distribution before and after oversampling to the console.</p>

  <h2>Script B: Feature Selection Methods</h2>
  <p><code>scriptb.py</code> implements feature selection methods to identify the most significant features for model training.</p>

  <h3>Dependencies</h3>
  <p>The script requires the following Python libraries:</p>
  <ul>
      <li>pandas</li>
      <li>numpy</li>
      <li>sklearn</li>
      <li>matplotlib (for visualization)</li>
  </ul>

  <h3>Usage</h3>
  <p>The dataset should be in a CSV file named <code>dataset.csv</code> in the same directory as the script. The script can be run with the following command:</p>
  <pre><code>python scriptb.py</code></pre>

  <h3>Functionality</h3>
  <p>The script performs the following steps:</p>
  <ol>
      <li>Reads the dataset and separates features from the target variable.</li>
      <li>Splits the data into training and test sets.</li>
      <li>Applies chi-squared feature selection to identify the top features.</li>
  </ol>

  <h3>Expected Outputs</h3>
  <p>The script outputs the indices of the selected features and may also provide visualizations of feature importances.</p>

</body>
</html>

