%% Intialize
clc; clear all;


%% Load Data
load('dataset.mat')

for i = 1:100
    if Y(i) == 0
        Y(i) = -1;
    end
end


%% Split data into test and train
row_idx = randperm(100);

X_Train_data = X(row_idx(1:50),:);
X_Test_data = X(row_idx(51:end),:);

Y_Train_data = Y(row_idx(1:50),:);
Y_Test_data = Y(row_idx(51:end),:);

%% Linear SVM 
       

[nsv, alpha, b0] = svc(X_Train_data,Y_Train_data,'linear');
       
svcoutput(X_Train_data,Y_Train_data,X_Test_data,'linear',alpha,b0);
       
Err_linear = svcerror(X_Train_data,Y_Train_data,X_Test_data,Y_Test_data,'linear',alpha,b0)
       
%% Polynomial SVM

Err_Poly = [];
optimal_finder = []

for sigma = 1:5
    
    global p1 p2;
    
    p1 = sigma;
    
    for c = 1:50
       [nsv, alpha, b0] = svc(X_Train_data,Y_Train_data,'poly',c);
       
       svcoutput(X_Train_data,Y_Train_data,X_Test_data,'poly',alpha,b0);
       
       Err_Poly = svcerror(X_Train_data,Y_Train_data,X_Test_data,Y_Test_data,'poly',alpha,b0);
       optimal_finder=[optimal_finder; p1, c, Err_Poly];
     
   end
end


       figure
       plot3(optimal_finder(:,1), optimal_finder(:,2), optimal_finder(:,3), 'mo');
       
       xlabel('Degree of the poylnomial')
       ylabel(' C')
       zlabel('Error')
       title('Polynomial Kernal')
       grid on
       axis('equal')
       
       poly_min_err = min(optimal_finder(:,3));
       index = find(optimal_finder(:,3)==poly_min_err);
       poly_min_err_finder = optimal_finder(index,:);
       
       hold on
       scatter3(poly_min_err_finder(:,1), poly_min_err_finder(:,2), poly_min_err_finder(:,3), 'filled');
       xlabel("Degree of the poylnomial");
       ylabel("C");
       zlabel("Min Error")
       title('Polynomial Kernal')
       grid on


%% RBF Kernel
Err_RBF = [];
rbf_optimal_finder = []

for sigma = 1:20
    
    global p1 p2;
    
    p1 = sigma;
    
    for c = 1:30
       [nsv, alpha, b0] = svc(X_Train_data,Y_Train_data,'rbf',c);
       
       svcoutput(X_Train_data,Y_Train_data,X_Test_data,'rbf',alpha,b0);
       
       Err_RBF= svcerror(X_Train_data,Y_Train_data,X_Test_data,Y_Test_data,'rbf',alpha,b0);
       rbf_optimal_finder=[rbf_optimal_finder; p1, c, Err_RBF];
     
   end
end



       figure
       plot3(rbf_optimal_finder(:,1), rbf_optimal_finder(:,2), rbf_optimal_finder(:,3), 'mo'); 
       xlabel('Sigma')
       ylabel('C')
       zlabel('Error')
       title('RBF Kernal')
       grid on
       axis('equal')
       
       RBF_min_err=min(rbf_optimal_finder(:,3));
       index = find(rbf_optimal_finder(:,3)==RBF_min_err);
       rbf_min_err_finder = rbf_optimal_finder(index,:);
       
       hold on
       scatter3(rbf_min_err_finder(:,1), rbf_min_err_finder(:,2), rbf_min_err_finder(:,3), 'filled');
       xlabel("Sigma");
       ylabel("C");
       zlabel("Min Error")
       title('RBF Kernal')
       grid on

      
fprintf('\n Linear loss value = %d', Err_linear)
       
fprintf('\n Polynomial least loss value = %d', poly_min_err)

fprintf('\n Polynomial Kernel Parameters ')
poly_min_err_finder

fprintf('\n RBF least loss value = %d',RBF_min_err)

fprintf('\n RBF Kernel Parameters ')
rbf_min_err_finder


