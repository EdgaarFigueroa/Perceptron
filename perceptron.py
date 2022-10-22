"""
    @Autor:         Varillas Figueroa Edgar Josue
    @Date:          21-10-2022
    @Description:   The implementation of a perceptron is carried
                    out considering the step function and the sigmoid 
                    function as activation functions to obtain the perceptron output.
                    The implementation consists of 2 exercises:
                        Exercise 1:
                            Has as inputs X=[0.4,0.3,0.2], weights
                            W=[0.9,0.4,0.3] and b=0
                            a) Implementing the perceptron with the step function
                            b) Implementing the perceptron with the sigmoid function
                        Exercise 2:
                            Has as inputs X=[0.4,0.3,0.2], weights
                            W=[0.9,0.4,0.3], b=0 and expected output y=0.2
                            a) Implementing the perceptron with the sigmoid function and
                            calculate the predicted output
                            b)Calculate the residual error
                            c)Calculate the SSe
"""

import numpy as np

def stepFunction(h):
    if h<=0:
        return 0
    else:
        return 1

def sigmoidFunction(h):
    return 1/(1+np.exp(-h))

def hFunction(X,W,b): #h=WX+b where W and X can be either scalar values or arrays
    return sum(np.multiply(X,W))+b

def residualError(predicted_output,expected_output): #e=y^-y where y^ and y can be either scalar values or arrays
    return predicted_output-expected_output

def SumOfSquaredError_vector(predicted_output,expected_output): #SSE= sum((y^-y)^2) where y^ and y must be arrays
    return sum((predicted_output-expected_output)**2)

def SumOfSquaredError(predicted_output,expected_output): #SSE= sum((y^-y)^2) where y^ and y must be scalar values
    return (predicted_output-expected_output)**2

def exercise1():
    print("Answers to exercise 1")
    # a)  
    print("Responses to item a")
    X=np.array([0.4,0.3,0.2])
    W=np.array([0.9,0.4,0.3])
    b=0
    h=hFunction(X,W,b)
    output=stepFunction(h)
    print(f"The output's perceptron is: {output}\n")

    # b)
    print("Responses to item b")
    output=sigmoidFunction(h)
    print(f"The output's perceptron is: {output}\n")

def exercise2():
    print("Answers to exercise 2")
    X=np.array([0.2,0.5,0.6])
    W=np.array([0.6,0.7,0.2])
    expected_output=0.2
    b=0
    h=hFunction(X,W,b)
    predicted_output=sigmoidFunction(h)
    residual_error=residualError(predicted_output,expected_output)
    SSE=SumOfSquaredError(predicted_output,expected_output)
    print(f"The predicted output's perceptron is: {predicted_output}")
    print(f"The residual error's perceptron is: {residual_error}")
    print(f"The SSE's perceptron is: {SSE}")

if __name__=="__main__":
    exercise1()
    exercise2()
