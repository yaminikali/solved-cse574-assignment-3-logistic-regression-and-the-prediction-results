Download Link: https://assignmentchef.com/product/solved-cse574-assignment-3-logistic-regression-and-the-prediction-results
<br>
<h1></h1>

We will evaluate your code by executing script.py file, which will internally call the problem specific functions. You must submit an assignment report (pdf file) summarizing your findings. In the problem statements below, the portions under REPORT heading need to be discussed in the assignment report.

Data Sets

In this assignment, we still use MNIST. In the script file provided to you, we have implemented a function, called <em>preprocess()</em>, with preprocessing steps. This will apply feature selection, feature normalization, and divide the dataset into 3 parts: training set, validation set, and testing set.

Your tasks

<ul>

 <li>Implement <strong>Logistic Regression </strong>and give the prediction results.</li>

 <li>Use the <strong>Support Vector Machine (SVM) </strong>toolbox svm.SVM to perform classification.</li>

 <li>Write a report to explain the experimental results with these 2 methods.</li>

 <li><em>Extra credit</em>: Implement the gradient descent minimization of multi-class <strong>Logistic Regression </strong>(using softmax function).</li>

</ul>

<h2>1.1         Problem 1: Implementation of Logistic Regression (40 code + 15 report = 55 points)</h2>

You are asked to implement Logistic Regression to classify hand-written digit images into correct corresponding labels. The data is the same that was used for the second programming assignment. Since the labels associated with each digit can take one out of 10 possible values (multiple classes), we cannot directly use a binary logistic regression classifier. Instead, we employ the <em>one-vs-all </em>strategy. In particular, you have to build 10 binary-classifiers (one for each class) to distinguish a given class from all other classes.

<h3>1.1.1         Implement <em>blrObjFunction() </em>function (20 points)</h3>

In order to implement Logistic Regression, you have to complete function <em>blrObjFunction() </em>provided in the base code (<em>script.py</em>). The input of <em>blrObjFunction.m </em>includes 3 parameters:

<ul>

 <li><strong>X </strong>is a data matrix where each row contains a feature vector in original coordinate (not including the bias 1 at the beginning of vector). In other words, <strong>X </strong>∈ R<em><sup>N</sup></em><sup>×<em>D</em></sup>. So you have to add the bias into each feature vector inside this function. In order to guarantee the consistency in the code and utilize automatic grading, <strong>please add the bias at the beginning of feature vector instead of the end</strong>. <strong>w</strong><em><sub>k </sub></em>is a column vector representing the parameters of Logistic Regression. Size of <strong>w</strong><em><sub>k </sub></em>is (<em>D </em>+ 1) × 1.</li>

 <li><strong>y</strong><em><sub>k </sub></em>is a column vector representing the labels of corresponding feature vectors in data matrix <strong>X</strong>. Each entry in this vector is either 1 or 0 to represent whether the feature vector belongs to a class <em>C<sub>k </sub></em>or not (<em>k </em>= 0<em>,</em>1<em>,</em>·· <em>,K </em>− 1). Size of <strong>y</strong><em><sub>k </sub></em>is <em>N </em>× 1 where <em>N </em>is the number of rows of <strong>X</strong>. The creation of <strong>y</strong><em><sub>k </sub></em>is already done in the base code.</li>

</ul>

Function <em>blrObjFunction() </em>has 2 outputs:

<ul>

 <li><strong>error </strong>is a scalar value which is the result of computing equation (2)</li>

 <li><strong>error </strong><strong>grad </strong>is a column vector of size (<em>D </em>+ 1) × 1 which represents the gradient of error function obtained by using equation (3).</li>

</ul>

<h3>1.1.2         Implement <em>blrPredict() </em>function</h3>

For prediction using Logistic Regression, given 10 weight vectors of 10 classes, we need to classify a feature vector into a certain class. In order to do so, given a feature vector <strong>x</strong>, we need to compute the posterior probability <em>P</em>(<em>y </em>= <em>C<sub>k</sub></em>|<strong>x</strong>) and the decision rule is to assign <strong>x </strong>to class <em>C<sub>k </sub></em>that maximizes <em>P</em>(<em>y </em>= <em>C<sub>k</sub></em>|<strong>x</strong>). In particular, you have to complete the function <em>blrPredict() </em>which returns the predicted label for each feature vector. Concretely, the input of <em>blrPredict() </em>includes 2 parameters:

<ul>

 <li>Similar to function <em>blrObjFunction()</em>, <strong>X </strong>is also a data matrix where each row contains a feature vector in original coordinate (not including the bias 1 at the beginning of vector). In other words, <strong>X </strong>has size <em>N </em>× <em>D</em>. In order to guarantee the consistency in the code and utilize automatic grading, <strong>please add the bias at the beginning of feature vector instead of the end</strong>.</li>

 <li><strong>W </strong>is a matrix where each column is a weight vector (<strong>w</strong><em><sub>k</sub></em>) of classifier for digit <em>k</em>. Concretely, <strong>W </strong>has size (<em>D </em>+ 1) × <em>K </em>where <em>K </em>= 10 is the number of classifiers.</li>

</ul>

The output of function <em>blrPredict() </em>is a column vector <strong>label </strong>which has size <em>N </em>× 1.

<h3>1.1.3         Report</h3>

In your report, you should train the logistic regressor using the given data <strong>X </strong>(Preprocessed feature vectors of MNIST data) with labels <strong>y</strong>. Record the total error with respect to each category in both training data and test data. And discuss the results in your report and explain why there is a difference between training error and test error.

<h2>1.2         For Extra Credit: Multi-class Logistic Regression</h2>

In this part, you are asked to implement multi-class Logistic Regression. Traditionally, Logistic Regression is used for binary classification. However, Logistic Regression can also be extended to solve the multi-class classification. With this method, we don’t need to build 10 classifiers like before. Instead, we now only need to build 1 classifier that can classify 10 classes at the same time.

<h3>1.2.1         Implement <em>mlrObjFunction() </em>function (10 points)</h3>

In order to implement Multi-class Logistic Regression, you have to complete function <em>mlrObjFunction() </em>provided in the base code (<em>script.py</em>). The input of <em>mlrObjFunction.m </em>includes the same definition of parameter as above. Function <em>mlrObjFunction() </em>has 2 outputs that has the same definition as above. You should use multi-class logistic function to regress the probability of each class.

<h3>1.2.2         Report</h3>

In your report, you should train the logistic regressor using the given data <strong>X</strong>(Preprocessed feature vectors of MNIST data) with labels <strong>y</strong>. Record the total error with respect to each category in both training data and test data. And discuss the results in your report and explain why there is a difference between training error and test error. Compare the performance difference between multi-class strategy with <em>one-vs-all </em>strategy.

<h2>1.3         Support Vector Machines</h2>

<h2>In this part of assignment you are asked to use the Support Vector Machine tool in sklearn.svm.SVM to perform classification on our data set. The details about the tool are provided here: <a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">http://scikit-learn. </a><a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">org/stable/modules/generated/sklearn.svm.SVC.html</a><a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">.</a></h2>

<h3>1.3.1         Implement <em>script.py </em>function)</h3>

Your task is to fill the code in Support Vector Machine section of <em>script.py </em>to learn the SVM model. The SVM models are known to be difficult to scale well to a large dataset. Please randomly sample 10<em>,</em>000 training samples to learn the SVM models, and compute accuracy of prediction with respect to training data, validation data and testing using the following parameters:

<ul>

 <li>Using linear kernel (all other parameters are kept default).</li>

 <li>Using radial basis function with value of gamma setting to 1 (all other parameters are kept default).</li>

 <li>Using radial basis function with value of gamma setting to default (all other parameters are kept default).</li>

 <li>Using radial basis function with value of gamma setting to default and varying value of C (1<em>,</em>10<em>,</em>20<em>,</em>30<em>,</em>·· <em>,</em>100) and plot the graph of accuracy with respect to values of C in the report.</li>

</ul>

After those experiments, choose the best choice of parameters, and train with the whole training dataset and report the accuracy in training, validation and testing data.

<h3>1.3.2         Report</h3>

In your report, you should train the SVM using the given data <strong>X</strong>(Preprocessed feature vectors of MNIST data) with labels <strong>y</strong>. And discuss the performance differences between linear kernel and radial basis, different gamma setting.

<h1>Appendices</h1>

<h2>A        Logistic Regression</h2>

Consider <strong>x </strong>∈ R<em><sup>D </sup></em>as an input vector. We want to classify <strong>x </strong>into correct class <em>C</em><sub>1 </sub>or <em>C</em><sub>2 </sub>(denoted as a random variable <em>y</em>). In Logistic Regression, the posterior probability of class <em>C</em><sub>1 </sub>can be written as follow:

<em>P</em>(<em>y </em>= <em>C</em><sub>1</sub>|<strong>x</strong>) = <em>σ</em>(<strong>w</strong><em><sup>T</sup></em><strong>x </strong>+ <em>w</em><sub>0</sub>)

where <strong>w </strong>∈ R<em><sup>D </sup></em>is the weight vector.

For simplicity, we will denote <strong>x </strong>= [1<em>,x</em><sub>1</sub><em>,x</em><sub>2</sub><em>,</em>··· <em>,x<sub>D</sub></em>] and <strong>w </strong>= [<em>w</em><sub>0</sub><em>,w</em><sub>1</sub><em>,w</em><sub>2</sub><em>,</em>··· <em>,w<sub>D</sub></em>]. With this new notation, the posterior probability of class <em>C</em><sub>1 </sub>can be rewritten as follow:

<em>P</em>(<em>y </em>= <em>C</em><sub>1</sub>|<strong>x</strong>) = <em>σ</em>(<strong>w</strong><em><sup>T</sup></em><strong>x</strong>)                                                                                (1)

And posterior probability of class <em>C</em><sub>2 </sub>is:

<em>P</em>(<em>y </em>= <em>C</em><sub>2</sub>|<strong>x</strong>) = 1 − <em>P</em>(<em>y </em>= <em>C</em><sub>1</sub>|<strong>x</strong>)

We now consider the data set {<strong>x</strong><sub>1</sub><em>,</em><strong>x</strong><sub>2</sub><em>,</em>··· <em>,</em><strong>x</strong><em><sub>N</sub></em>} and corresponding label {<em>y</em><sub>1</sub><em>,y</em><sub>2</sub><em>,</em>··· <em>,y<sub>N</sub></em>} where

1           if <strong>x</strong><em><sub>i </sub></em>∈ <em>C</em><sub>1</sub>

0           if <strong>x</strong><em><sub>i </sub></em>∈ <em>C</em><sub>2</sub>

for <em>i </em>= 1<em>,</em>2<em>,</em>··· <em>,N</em>.

With this data set, the likelihood function can be written as follow:

where <em>θ<sub>n </sub></em>= <em>σ</em>(<strong>w</strong><em><sup>T</sup></em><strong>x</strong><em><sub>n</sub></em>) for <em>n </em>= 1<em>,</em>2<em>,</em>··· <em>,N</em>.

We also define the error function by taking the negative logarithm of the log likelihood, which gives the cross-entropy error function of the form:

(2)

Note that this function is different from the squared loss function that we have used for Neural Networks and Perceptrons.

The gradient of error function with respect to <strong>w </strong>can be obtained as follow:

(3)

Up to this point, we can use again gradient descent to find the optimal weight <strong>w </strong>to minimize the error b

function with the formula:

<strong>w</strong><em>new </em>= <strong>w</strong><em>old </em>− <em>η</em>∇<em>E</em>(<strong>w</strong><em>old</em>)                                                                              (4)

<h2>B          Multi-Class Logistic Regression</h2>

For multi-class Logistic Regression, the posterior probabilities are given by a softmax transformation of linear functions of the feature variables, so that

<em>P</em>(<em>y </em>= <em>C</em><em>k</em>|<strong>x</strong>) = <sub>P</sub><em>j </em>exp(<strong>w</strong><em>T<sub>j </sub></em><strong>x</strong>)                                                              (5)

Now we write down the likelihood function. This is most easily done using the 1-of-K coding scheme in which the target vector <strong>y</strong><em><sub>n </sub></em>for a feature vector <strong>x</strong><em><sub>n </sub></em>belonging to class <em>C<sub>k </sub></em>is a binary vector with all elements zero except for element <em>k</em>, which equals one. The likelihood function is then given by

<table width="0">

 <tbody>

  <tr>

   <td width="167">                                           <em>N             K</em><em>P</em>(<strong>Y</strong>|<strong>w</strong><sub>1</sub><em>,</em>··· <em>,</em><strong>w</strong><em><sub>K</sub></em>) = <sup>YY</sup><em>n</em>=1<em>k</em>=1</td>

   <td width="303">                                         <em>N         K</em><em>P</em>(<em>y </em>= <em>C</em><em>k</em>|<strong>x</strong><em>n</em>)<em>y</em><em>nk </em>= YY <em>θ</em><em>nky</em><em>nk</em><em>n</em>=1<em>k</em>=1</td>

   <td width="17">(6)</td>

  </tr>

 </tbody>

</table>

where <em>θ<sub>nk </sub></em>is given by (5) and <em>Y </em>is an <em>N </em>× <em>K </em>matrix (obtained using 1-of-K encoding) of target variables with elements <em>y<sub>nk</sub></em>. Taking the negative logarithm then gives

<em>N         K</em>

<em>E</em>(<strong>w</strong><sub>1</sub><em>,</em>··· <em>,</em><strong>w</strong><em><sub>K</sub></em>) = −ln <em>P</em>(<strong>Y</strong>|<strong>w</strong><sub>1</sub><em>,</em>··· <em>,</em><strong>w</strong><em><sub>K</sub></em>) = − <sup>XX</sup><em>y<sub>nk </sub></em>ln <em>θ<sub>nk                                                                                  </sub></em>(7)

<em>n</em>=1<em>k</em>=1

which is known as the cross-entropy error function for the multi-class classification problem.

We now take the gradient of the error function with respect to one of the parameter vectors <strong>w</strong><em><sub>k </sub></em>. Making use of the result for the derivatives of the softmax function, we obtain:

<em>∂E</em>(<strong>w</strong><sub>1</sub><em>,</em>··· <em>,</em><strong>w</strong><em><sub>K</sub></em>) <sub>X</sub><em><sup>N</sup></em>

=           (<em>θ<sub>nk </sub></em>− <em>y<sub>nk</sub></em>)<strong>x</strong><em><sub>n                                                                                                  </sub></em>(8)

<em>∂</em><strong>w</strong><em><sub>k </sub></em><em>n</em>=1

then we could use the following updating function to get the optimal parameter vector w iteratively: