% This is the code of the case study conducted in the paper "A multi-objective 
% evolutionary approach towards automatedonline controlled experiments"
%
%

clear all;
% set these variables to switch between different experimental scenarios
max_line_num = -1;
should_train = false;
should_test = false;
exp_setting = 6;

% below variables will be saved and loaded in testing
global x_c;
global alpha_start;
global within_range_th;
global population_size; % used only in moea function
global iteration_num;
global num_var; 
global num_curves; 
global algo_names;
global algo_handlers;
global population_sizes; 
global num_obj;
global obj_fcns;
global is_soga;

alpha_start = 1;
within_range_th = 2;
population_size = 100;

iteration_num = 100;
num_var=3;
is_soga=false;
% TODO: use smaller range so that result could be e.g. -20% instead of -100%
x_c = [randi([-150 150]),randi([-150 150]),randi([-150 150])];

if exp_setting==1
    % RQ1
    num_curves=5;
    num_obj = {3,3,3,3,3};
    population_sizes = {100,100,100,100,100};
    algo_names = {'NSGAII','SPEA2','PESAII','MOEAD','AGEII'};
    algo_handlers = {@NSGAII,@SPEA2,@PESAII,@MOEAD,@AGEII};
    obj_fcns = {{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric},{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric},{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric},{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric},{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric}};
    within_range_th = 0.05;
elseif exp_setting==2
    % RQ2: SOGA using only one weighted objective(activeTimeOsAndroid)
    num_curves=3;
    num_obj = {1,1,1};
    population_sizes = {100,100,100};
    algo_names = {'GA','DE','SA'};
    algo_handlers = {@GA,@DE,@SA}; 
    obj_fcns = {{@f_activeTimeOsAndroidMetric},{@f_activeTimeOsAndroidMetric},{@f_activeTimeOsAndroidMetric}};
    within_range_th = 0.05;
    is_soga=true;
elseif exp_setting==3
    % RQ2: SOGA using sum of equal weighted objectives on 3 metrics.
    num_curves=3;
    num_obj = {1,1,1};
    population_sizes = {100,100,100};
    algo_names = {'GA','DE','SA'}; %'PSO' is removed due to a weird issue. Maybe there is a bug in PSO implementation.
    algo_handlers = {@GA,@DE,@SA};
    obj_fcns = {{@f_activeTimeOsMetricEqualWeightedSumSOGA},{@f_activeTimeOsMetricEqualWeightedSumSOGA},{@f_activeTimeOsMetricEqualWeightedSumSOGA}};
    within_range_th = 0.05;
    is_soga=true;
elseif exp_setting==4
    % RQ3
    num_curves=8;
    num_obj = {3,3,3,3,3,3,3,3};
    population_sizes = {2,5,10,20,50,100,150,200};
    algo_names = {'SPEA2','SPEA2','SPEA2','SPEA2','SPEA2','SPEA2','SPEA2','SPEA2'};
    algo_handlers = {@SPEA2,@SPEA2,@SPEA2,@SPEA2,@SPEA2,@SPEA2,@SPEA2,@SPEA2};
    obj_fcns = {{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric},{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric},{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric},{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric},{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric},{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric},{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric},{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric}};
elseif exp_setting==5
    % RQ3
    name = 'MOEAD';
    handler = @MOEAD;
    num_curves=8;
    num_obj = {3,3,3,3,3,3,3,3};
    population_sizes = {20,30,50,80,100,150,200,300};
    algo_names = {name,name,name,name,name,name,name,name};
    algo_handlers = {handler,handler,handler,handler,handler,handler,handler,handler};
    obj_fcns = {{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric},{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric},{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric},{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric},{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric},{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric},{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric},{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric}};
elseif exp_setting==6
    % [for rapid debugging only]
    % RQ2: SOGA using sum of equal weighted objectives on 3 metrics.
    num_curves=1;
    num_obj = {1};
    population_sizes = {100};
    algo_names = {'GA'};
    algo_handlers = {@GA};
    obj_fcns = {{@f_activeTimeOsMetricEqualWeightedSumSOGA}};
    within_range_th = 0.05;
    is_soga=true;
    iteration_num = 50;
elseif exp_setting==7
    % [for rapid debugging only]
    % RQ2: SOGA using sum of equal weighted objectives on 3 metrics.
    num_curves=1;
    num_obj = {3};
    population_sizes = {100};
    algo_names = {'MOEAD'};
    algo_handlers = {@MOEAD};
    obj_fcns = {{@f_activeTimeOsAndroidMetric,@f_activeTimeOsWindowsMetric,@f_activeTimeOsMacMetric}};
    within_range_th = 0.05;
    iteration_num = 100;
end

save('initial_params.mat','x_c','alpha_start','within_range_th', 'population_size', 'population_sizes', 'iteration_num', 'algo_names','num_var','num_curves','algo_handlers','num_obj','obj_fcns'); %TODO:fix: it may override original read_test_data.mat

if should_train
    training_filename = 'D:\Study\Research\MOEA\AddressaDataset\three_month\d30_hotfixed_simple_model1231.npy';
    [training_coeffs,training_y_activeTime,training_y_location,training_y_session,training_y_os,training_y_deviceType,training_y_browser,training_y_sessionStart,training_y_sessionStop] = getYandCoeff(training_filename, max_line_num);
    
    % define multiple objectives used in training
    [training_y_activeTimeCityOslo,training_y_activeTimeCityTrondheim,training_y_activeTimeRegionOslo,training_y_activeTimeRegionSorTrondelag,training_y_activeTimeDeviceDesktop,training_y_activeTimeDeviceMobile,training_y_activeTimeDeviceTablet,training_y_activeTimeOsWindows,training_y_activeTimeOsAndroid,training_y_activeTimeOsMac,training_y_activeTimeOsIphone,training_y_activeTimeSessionStart] = getObjectiveYFromData(training_y_activeTime,training_y_location,training_y_deviceType,training_y_os,training_y_sessionStart);
   
    % define models optimized for multiple objectives
    global Mdl;global Mdl_activeTimeCityOslo;global Mdl_activeTimeCityTrondheim;global Mdl_activeTimeRegionOslo; global Mdl_activeTimeRegionSorTrondelag;global Mdl_activeTimeDeviceDesktop;global Mdl_activeTimeDeviceMobile;global Mdl_activeTimeDeviceTablet;global Mdl_activeTimeOsWindows;global Mdl_activeTimeOsAndroid;global Mdl_activeTimeOsMac;global Mdl_activeTimeOsIphone;global Mdl_activeTimeSessionStart;
    [Mdl,FitInfo] = fitrlinear(training_coeffs,training_y_activeTime);
    [Mdl_activeTimeCityOslo,FitInfo] = fitrlinear(training_coeffs,training_y_activeTimeCityOslo);
    [Mdl_activeTimeCityTrondheim,FitInfo] = fitrlinear(training_coeffs,training_y_activeTimeCityTrondheim);
    [Mdl_activeTimeRegionOslo,FitInfo] = fitrlinear(training_coeffs,training_y_activeTimeRegionOslo);
    [Mdl_activeTimeRegionSorTrondelag,FitInfo] = fitrlinear(training_coeffs,training_y_activeTimeRegionSorTrondelag);
    [Mdl_activeTimeDeviceDesktop,FitInfo] = fitrlinear(training_coeffs,training_y_activeTimeDeviceDesktop);
    [Mdl_activeTimeDeviceMobile,FitInfo] = fitrlinear(training_coeffs,training_y_activeTimeDeviceMobile);
    [Mdl_activeTimeDeviceTablet,FitInfo] = fitrlinear(training_coeffs,training_y_activeTimeDeviceTablet);
    [Mdl_activeTimeOsWindows,FitInfo] = fitrlinear(training_coeffs,training_y_activeTimeOsWindows);
    [Mdl_activeTimeOsAndroid,FitInfo] = fitrlinear(training_coeffs,training_y_activeTimeOsAndroid);
    [Mdl_activeTimeOsMac,FitInfo] = fitrlinear(training_coeffs,training_y_activeTimeOsMac);
    [Mdl_activeTimeOsIphone,FitInfo] = fitrlinear(training_coeffs,training_y_activeTimeOsIphone);
    [Mdl_activeTimeSessionStart,FitInfo] = fitrlinear(training_coeffs,training_y_activeTimeSessionStart);
    
    save('input_data_and_fit_model.mat')
end

if should_test
    load('input_data_and_fit_model.mat')

    global testing_coeffs;global testing_y_activeTime;

    test_file = 'D:\Study\Research\MOEA\AddressaDataset\three_month\d30_coldfixed_simple_model1231.npy'; 
    [testing_coeffs,testing_y_activeTime,testing_y_location,testing_y_session,testing_y_os,testing_y_deviceType,testing_y_browser,testing_y_sessionStart,testing_y_sessionStop] = getYandCoeff(test_file, max_line_num);
    
    % define multiple objectives used in MOEA
    global testing_y_activeTimeCityOslo;global testing_y_activeTimeCityTrondheim;global testing_y_activeTimeRegionOslo;global testing_y_activeTimeRegionSorTrondelag;global testing_y_activeTimeDeviceDesktop;global testing_y_activeTimeDeviceMobile;global testing_y_activeTimeDeviceTablet;global testing_y_activeTimeOsWindows;global testing_y_activeTimeOsAndroid;global testing_y_activeTimeOsMac;global testing_y_activeTimeOsIphone;global testing_y_activeTimeSessionStart;
    [testing_y_activeTimeCityOslo,testing_y_activeTimeCityTrondheim,testing_y_activeTimeRegionOslo,testing_y_activeTimeRegionSorTrondelag,testing_y_activeTimeDeviceDesktop,testing_y_activeTimeDeviceMobile,testing_y_activeTimeDeviceTablet,testing_y_activeTimeOsWindows,testing_y_activeTimeOsAndroid,testing_y_activeTimeOsMac,testing_y_activeTimeOsIphone,testing_y_activeTimeSessionStart] = getObjectiveYFromData(testing_y_activeTime, testing_y_location,testing_y_deviceType,testing_y_os,testing_y_sessionStart)

    % calculate the loss between predicted model and groundtruth
    L_train = loss(Mdl, training_coeffs, training_y_activeTime);
    L_test = loss(Mdl, testing_coeffs, testing_y_activeTime);
    
    L_train_activeTimeCityOslo = loss(Mdl_activeTimeCityOslo, training_coeffs, training_y_activeTimeCityOslo);
    L_test_activeTimeCityOslo = loss(Mdl_activeTimeCityOslo, testing_coeffs, testing_y_activeTimeCityOslo);
    
    % below code compares the current model with the loss of random model for verification purpose
    % [MdlRand,FitInfoRand] = fitrlinear(training_coeffs,rand(1001,1));
    % L_train_rand = loss(MdlRand, training_coeffs, training_y_activeTime);
    % L_test_rand = loss(MdlRand, testing_coeffs, testing_y_activeTime);
    
    save('read_test_data.mat');
else
    % it may include variables that have old/outdated values such as num_var etc.
    load('read_test_data.mat'); 
end

% load several variables defined at first to overwrite old values from read_test_data.mat
load('initial_params.mat'); 

NumPFSolutionsAcc=cell(1, num_curves);
NumWithInRangeSolutionsAcc=cell(1, num_curves);
NumPFSolutions=cell(1, num_curves);
NumWithInRangeSolutions=cell(1, num_curves);
NumRawSolutions=cell(1, num_curves);
AvgDistToGroundTruthAcc=cell(1, num_curves);
AvgDistToGroundTruth=cell(1, num_curves);
MinDistToGroundTruthAcc=cell(1, num_curves);
MinDistToGroundTruth=cell(1, num_curves);
AvgMetricResultAcc=cell(1, num_curves);
AvgMetricResult=cell(1, num_curves);

% before perform MOEA algos, first delete Data/algo_names{i} foldes from previous runs
for i=1:num_curves
    [status,message,messageid] = rmdir("Data/"+algo_names{i},'s');
end

file_num = 1;
for i=1:num_curves
    platemo('algorithm',algo_handlers{i}, 'N',population_sizes{i},'objFcn',obj_fcns{i}, 'upper',zeros(1,num_var)+10, 'lower',zeros(1,num_var)-10,'D',num_var,'M',num_obj{i},'save',iteration_num,'maxFE',population_sizes{i} * iteration_num);
    if (i>1 && strcmp(algo_names{i-1},algo_names{i}))
        file_num = file_num + 1;
    end
    load("Data/"+algo_names{i}+"/"+algo_names{i}+"_UserProblem_M"+num_obj{i}+"_D"+num_var+"_"+file_num+".mat", 'result');
    [NumPFSolutionsAcc{i},NumWithInRangeSolutionsAcc{i},NumPFSolutions{i},NumWithInRangeSolutions{i},NumRawSolutions{i},AvgDistToGroundTruthAcc{i},AvgDistToGroundTruth{i},MinDistToGroundTruthAcc{i},MinDistToGroundTruth{i},AvgMetricResultAcc{i},AvgMetricResult{i}] = plot_moea_result(result, num_obj{i});
end

curve_names=cell(1, num_curves);
for i=1:num_curves
    curve_names{i}=algo_names{i}+"-"+population_sizes{i};
end

figure;
for i=1:num_curves
    plot(1:iteration_num, NumPFSolutionsAcc{i});
    disp('NumPFSolutionsAcc ' + curve_names{i});
    disp(NumPFSolutionsAcc{i}(iteration_num));
    hold on
end
legend(curve_names);
xlabel('Iterations');
ylabel('# Pareto Front Solutions');

figure;
for i=1:num_curves
    plot(1:iteration_num, NumWithInRangeSolutionsAcc{i});
    hold on
end
legend(curve_names);
xlabel('Iterations');
ylabel('# With-in Range Solutions');

figure;
for i=1:num_curves
    plot(1:iteration_num, abs(NumWithInRangeSolutionsAcc{i}) ./ max(1,NumPFSolutionsAcc{i})); % if denominator is 0, use 1 to avoid devidedByZero error. 
    hold on
end
legend(curve_names);
xlabel('Iterations');
ylabel('Percentage of With-in Range Solutions');

figure;
for i=1:num_curves
    plot(1:iteration_num, NumRawSolutions{i});
    disp('NumRawSolutions ' + curve_names{i});
    disp(NumRawSolutions{i}(iteration_num));
    hold on
end
legend(curve_names);
xlabel('Iterations');
ylabel('# Raw Solutions Per Iteration');

figure;
for i=1:num_curves
    plot(1:iteration_num, NumPFSolutions{i});
    disp('NumPFSolutions ' + curve_names{i});
    disp(NumPFSolutions{i}(iteration_num));
    hold on
end
legend(curve_names);
xlabel('Iterations');
ylabel('# Pareto Front Solutions Per Iteration');

figure;
for i=1:num_curves
    plot(1:iteration_num, NumWithInRangeSolutions{i});
    hold on
end
legend(curve_names);
xlabel('Iterations');
ylabel('# With-in Range Solutions Per Iteration');

figure;
for i=1:num_curves
    plot(1:iteration_num, AvgDistToGroundTruthAcc{i});
    disp('AvgDistToGroundTruthAcc ' + curve_names{i});
    disp(AvgDistToGroundTruthAcc{i}(iteration_num));
    hold on
end
legend(curve_names);
xlabel('Iterations');
ylabel('Average Distance to Ground Truth');
 
figure;
for i=1:num_curves
    plot(1:iteration_num, AvgDistToGroundTruth{i});
    disp('AvgDistToGroundTruth ' + curve_names{i});
    disp(AvgDistToGroundTruth{i}(iteration_num));
    hold on
end
legend(curve_names);
xlabel('Iterations');
ylabel('Average Distance to Ground Truth Per Iteration');

top_k = [1,3,5,10];
for min_dist_row_idx=1:4
    figure;
    for i=1:num_curves
        plot(1:iteration_num, MinDistToGroundTruthAcc{i}(min_dist_row_idx,:));
        disp('MinDistToGroundTruthAcc ' + curve_names{i} + ' top ' + num2str(top_k(min_dist_row_idx)));
        disp(MinDistToGroundTruthAcc{i}(iteration_num));
        hold on
    end
    legend(curve_names);
    xlabel('Iterations');
    ylabel(['Top ' num2str(top_k(min_dist_row_idx)) ' Min Distance to Ground Truth']);

    figure;
    for i=1:num_curves
        plot(1:iteration_num, MinDistToGroundTruth{i}(min_dist_row_idx,:));
        disp('MinDistToGroundTruth ' + curve_names{i} + ' top ' + num2str(top_k(min_dist_row_idx)));
        disp(MinDistToGroundTruth{i}(iteration_num));
        hold on
    end
    legend(curve_names);
    xlabel('Iterations');
    ylabel(['Top ' num2str(top_k(min_dist_row_idx)) ' Min Distance to Ground Truth Per Iteration']);
end

for metric_idx=1:3
    figure;
    for i=1:num_curves
        plot(1:iteration_num, AvgMetricResultAcc{i}(metric_idx,:));
        disp('AvgMetricResultAcc ' + curve_names{i} + ' metric_' + num2str(metric_idx));
        disp(AvgMetricResultAcc{i}(metric_idx, iteration_num));
        hold on
    end
    legend(curve_names);
    xlabel('Iterations');
    ylabel(['Avg % change on metric ' num2str(metric_idx)]);
    
    figure;
    for i=1:num_curves
        plot(1:iteration_num, AvgMetricResult{i}(metric_idx,:));
        disp('AvgMetricResult ' + curve_names{i} + ' metric_' + num2str(metric_idx));
        disp(AvgMetricResult{i}(metric_idx, iteration_num));
        hold on
    end
    legend(curve_names);
    xlabel('Iterations');
    ylabel(['Avg % change on metric ' num2str(metric_idx) ' Per Iteration']);
end

function [y_activeTimeCityOslo,y_activeTimeCityTrondheim,y_activeTimeRegionOslo,y_activeTimeRegionSorTrondelag,y_activeTimeDeviceDesktop,y_activeTimeDeviceMobile,y_activeTimeDeviceTablet,y_activeTimeOsWindows,y_activeTimeOsAndroid,y_activeTimeOsMac,y_activeTimeOsIphone,y_activeTimeSessionStart] = getObjectiveYFromData(y_activeTime,y_location,y_deviceType,y_os,y_sessionStart)
    y_activeTimeCityOslo = y_activeTime;
    y_activeTimeCityTrondheim = y_activeTime;
    y_activeTimeRegionOslo = y_activeTime;
    y_activeTimeRegionSorTrondelag = y_activeTime;
    y_activeTimeDeviceDesktop = y_activeTime;
    y_activeTimeDeviceMobile = y_activeTime;
    y_activeTimeDeviceTablet = y_activeTime;
    y_activeTimeOsWindows = y_activeTime;
    y_activeTimeOsAndroid = y_activeTime;
    y_activeTimeOsMac = y_activeTime;
    y_activeTimeOsIphone = y_activeTime;
    y_activeTimeSessionStart = y_activeTime;
    % TODO: change activeTime to view by doing view= (activeTime > 0) ? 1 : 0
    for i = 1:length(y_activeTime)
        if ~contains(y_location(i),"_oslo_")
            y_activeTimeCityOslo(i) = 0;
        end
        if ~contains(y_location(i),"_trondheim_")
            y_activeTimeCityTrondheim(i) = 0;
        end
        if ~endsWith(y_location(i),"_oslo")
            y_activeTimeRegionOslo(i) = 0;
        end
        if ~endsWith(y_location(i),"_sor-trondelag")
            y_activeTimeRegionSorTrondelag(i) = 0;
        end
        if ~strcmp(y_deviceType(i),"Desktop")
            y_activeTimeDeviceDesktop(i) = 0;
        end
        if ~strcmp(y_deviceType(i),"Mobile")
            y_activeTimeDeviceMobile(i) = 0;
        end
        if ~strcmp(y_deviceType(i),"Tablet")
            y_activeTimeDeviceTablet(i) = 0;
        end
        if ~strcmp(y_os(i),"Windows")
            y_activeTimeOsWindows(i) = 0;
        end
        if ~strcmp(y_os(i),"Android")
            y_activeTimeOsAndroid(i) = 0;
        end
        if ~strcmp(y_os(i),"Macintosh")
            y_activeTimeOsMac(i) = 0;
        end
        if ~strcmp(y_os(i),"iPhone OS")
            y_activeTimeOsIphone(i) = 0;
        end
        if ~strcmp(y_sessionStart(i),"True")
            y_activeTimeSessionStart(i) = 0;
        end
    end
end

function atcl_vec = get_atcl_vec(article)
    atcl_vec = [];
    for i =1 : length(article)-8
        value = str2double(article(i));
        atcl_vec = [atcl_vec; value];
    end
end

% An example of the output:
%{... atcl floats... '0.05918367346938776' (activeTime);'no_trondheim_sor-trondelag';'20';'Android';'Mobile';'';'False';'False'}
function [activeTime,location,session,os,deviceType,browser,sessionStart,sessionStop] = get_y_from_atcl(article)
    activeTime = str2double(article(length(article)-7));
    location = article(length(article)-6);
    session = article(length(article)-5);
    os = article(length(article)-4);
	deviceType = article(length(article)-3);
    browser = article(length(article)-2);
    sessionStart = article(length(article)-1);
    sessionStop = article(length(article));
end

function [coefficients, y_activeTime, y_location, y_session, y_os, y_deviceType, y_browser, y_sessionStart, y_sessionStop] = getYandCoeff(file_name,max_line_num)
    %process python data
    num_atcl = 30;
    beta = ones(1,num_atcl);

    coefficients = [];
    y_activeTime = [];
    y_location = [];
    y_session = [];
    y_os = [];
    y_deviceType = [];
    y_browser = [];
    y_sessionStart = [];
    y_sessionStop = [];
    
    fid = fopen(file_name);
    tline = fgetl(fid);
    while ischar(tline)
        if max_line_num > 0 && length(y_activeTime)>max_line_num 
            break; 
        end
        %disp(tline)
        articlesStrArray = split(tline,';');

        target_article = split(articlesStrArray(length(articlesStrArray)),',');
        target_atcl = get_atcl_vec(target_article);
        [activeTime,location,session,os,deviceType,browser,sessionStart,sessionStop] = get_y_from_atcl(target_article);
        y_activeTime = [y_activeTime;activeTime];
        y_location = [y_location;location];
        y_session = [y_session;session];
        y_os = [y_os;os];
        y_deviceType = [y_deviceType;deviceType];
        y_browser = [y_browser;browser];
        y_sessionStart = [y_sessionStart;sessionStart];
        y_sessionStop = [y_sessionStop;sessionStop];

        param = [];
        for i=1:length(articlesStrArray)-1
            atcl = split(articlesStrArray(i),',');
            atcl_vec = get_atcl_vec(atcl);
            param = [param, beta(i)*rot90(atcl_vec)*target_atcl];
        end
        coefficients = [coefficients; param];
        tline = fgetl(fid);
    end
    fclose(fid);
end

function metric = getActiveTimeMetric(x, testing_y_activeTime, model, start_idx, end_idx)
    metric = getActiveTimeMetricWithPrintInfo(x, testing_y_activeTime, model, start_idx, end_idx, false);
end

function metric = getActiveTimeMetricPrint(x, testing_y_activeTime, model, start_idx, end_idx)
    metric = getActiveTimeMetricWithPrintInfo(x, testing_y_activeTime, model, start_idx, end_idx, true);
end

function metric = getActiveTimeMetricWithPrintInfo(x, testing_y_activeTime, model, start_idx, end_idx, should_print)
    global testing_coeffs;global x_c;global alpha_start;
    
    alpha_v = model.Beta();
    alpha_c = model.Beta();
    alpha_v(alpha_start:alpha_start+end_idx-start_idx)=x(start_idx:end_idx);
    alpha_c(alpha_start:alpha_start+end_idx-start_idx)=x_c(start_idx:end_idx);% we change to sub-optimal parameters for control
    
    y_predict_v = testing_coeffs*alpha_v+ones(size(testing_coeffs, 1),1)*model.Bias;
    y_predict_c = testing_coeffs*alpha_c+ones(size(testing_coeffs, 1),1)*model.Bias;
    len = length(y_predict_c);
    y_is_v = mod(bsxfun(@plus,1,1:len),2)';
    y_is_c = 1 - y_is_v;
    metric_c = y_is_c .*(testing_y_activeTime - abs(y_predict_c - testing_y_activeTime));
    metric_v = y_is_v .*(testing_y_activeTime - abs(y_predict_v - testing_y_activeTime));
    metric = (-sum(metric_v)+sum(metric_c))/abs(sum(metric_c));
    %metric = rand() %why 1 pareto?
    if should_print
        disp("x="+x(start_idx:end_idx)+";alpha_c="+alpha_c(alpha_start+start_idx-1:alpha_start+end_idx-1)+";Mdl.Base="+model.Beta(alpha_start+start_idx-1:alpha_start+end_idx-1));
        disp("metric="+metric+";sum(metric_v)="+sum(metric_v)+"sum(metric_c)="+sum(metric_c));
    end
end

function metric = f_activeTimeMetric(x)
    global testing_y_activeTime; % groundtruth t (activeTime)
    global Mdl;
    metric = getActiveTimeMetric(x, testing_y_activeTime, Mdl, 1,1);
end

function metric = f_activeTimeMetricNeg(x)
    global testing_y_activeTime; % groundtruth t (activeTime)
    global Mdl;
    metric = -getActiveTimeMetric(x, testing_y_activeTime, Mdl,1,1);
end


function metric = f_activeTimeCityOMetric(x)
    global testing_y_activeTimeCityOslo;
    global Mdl_activeTimeCityOslo;
    metric = getActiveTimeMetric(x, testing_y_activeTimeCityOslo, Mdl_activeTimeCityOslo,1,1);
end

function metric = f_activeTimeCityTMetric(x)
    global testing_y_activeTimeCityTrondheim;
    global Mdl_activeTimeCityTrondheim;
    metric = getActiveTimeMetric(x, testing_y_activeTimeCityTrondheim,Mdl_activeTimeCityTrondheim,1,1);
end

function metric = f_activeTimeRegionOMetric(x)
    global testing_y_activeTimeRegionOslo;
    global Mdl_activeTimeRegionOslo;
    metric = getActiveTimeMetric(x, testing_y_activeTimeRegionOslo,Mdl_activeTimeRegionOslo,2,2);
end

function metric = f_activeTimeRegionSMetric(x)
    global testing_y_activeTimeRegionSorTrondelag;
    global Mdl_activeTimeRegionSorTrondelag;
    metric = getActiveTimeMetric(x, testing_y_activeTimeRegionSorTrondelag,Mdl_activeTimeRegionSorTrondelag,1,1);
end

function metric = f_activeTimeDeviceDesktopMetric(x)
    global testing_y_activeTimeDeviceDesktop;
    global Mdl_activeTimeDeviceDesktop;
    metric = getActiveTimeMetric(x, testing_y_activeTimeDeviceDesktop,Mdl_activeTimeDeviceDesktop,1,1);
end

function metric = f_activeTimeDeviceMobileMetric(x)
    global testing_y_activeTimeDeviceMobile;
    global Mdl_activeTimeDeviceMobile;
    metric = getActiveTimeMetric(x, testing_y_activeTimeDeviceMobile,Mdl_activeTimeDeviceMobile,1,1);
end

function metric = f_activeTimeDeviceTabletMetric(x)
    global testing_y_activeTimeDeviceTablet;
    global Mdl_activeTimeDeviceTablet;
    metric = getActiveTimeMetric(x, testing_y_activeTimeDeviceTablet,Mdl_activeTimeDeviceTablet,1,1);
end

function metric = f_activeTimeOsWindowsMetric(x)
    global testing_y_activeTimeOsWindows;
    global Mdl_activeTimeOsWindows;
    metric = getActiveTimeMetric(x, testing_y_activeTimeOsWindows,Mdl_activeTimeOsWindows,2,2);
end

function metric = f_activeTimeOsAndroidMetric(x)
    global testing_y_activeTimeOsAndroid;
    global Mdl_activeTimeOsAndroid;
    metric = getActiveTimeMetric(x, testing_y_activeTimeOsAndroid,Mdl_activeTimeOsAndroid,1,1);
end

function metric = f_activeTimeOsMacMetric(x)
    global testing_y_activeTimeOsMac;
    global Mdl_activeTimeOsMac;
    metric = getActiveTimeMetric(x, testing_y_activeTimeOsMac,Mdl_activeTimeOsMac,3,3);
end

function metric = f_activeTimeOsMetricEqualWeightedSumSOGA(x)
    metric = (f_activeTimeOsMacMetric(x)+f_activeTimeOsAndroidMetric(x)+f_activeTimeOsWindowsMetric(x))/3;
end

function [metric,osMacMetric,osAndroidMetric,osWindowsMetric] = f_activeTimeOsMetricEqualWeightedSumSOGAWithBreakdown(x)
    osMacMetric = f_activeTimeOsMacMetric(x);
    osAndroidMetric = f_activeTimeOsAndroidMetric(x);
    osWindowsMetric = f_activeTimeOsWindowsMetric(x);
    metric = (osMacMetric+osAndroidMetric+osWindowsMetric)/3;
end

function metric = f_activeTimeOsIphoneMetric(x)
    global testing_y_activeTimeOsIphone;
    global Mdl_activeTimeOsIphone;
    metric = getActiveTimeMetric(x, testing_y_activeTimeOsIphone,Mdl_activeTimeOsIphone,1,1);
end

function metric = f_activeTimeSessionStartMetric(x)
    global testing_y_activeTimeSessionStart;
    global Mdl_activeTimeSessionStart;
    metric = getActiveTimeMetric(x, testing_y_activeTimeSessionStart,Mdl_activeTimeSessionStart,1,1);
end


function [NumPFSolutionsAcc,NumWithInRangeSolutionsAcc,NumPFSolutions,NumWithInRangeSolutions,NumRawSolutions, AvgDistToGroundTruthAcc, AvgDistToGroundTruth, MinDistToGroundTruthAcc, MinDistToGroundTruth, AvgMetricResultAcc, AvgMetricResult] = plot_moea_result(result, num_obj)
    global iteration_num;
    % getGroundTruth
    global Mdl_activeTimeOsAndroid;global Mdl_activeTimeOsWindows;global Mdl_activeTimeOsMac;
    global alpha_start;
    global is_soga;
    
    b1=Mdl_activeTimeOsAndroid.Beta();
    b2=Mdl_activeTimeOsWindows.Beta();
    b3=Mdl_activeTimeOsMac.Beta();
    ground_truth = [b1(alpha_start:alpha_start),b2(alpha_start:alpha_start),b3(alpha_start:alpha_start)];

    PFSolutionsAcc = [];
    WithInRangeSolutionsAcc = [];
    
    % TODO: replace it with zeros, same below
    NumPFSolutionsAcc = size(1,iteration_num); 
    NumWithInRangeSolutionsAcc = size(1,iteration_num);
    NumPFSolutions = size(1,iteration_num);
    NumWithInRangeSolutions = size(1,iteration_num);
    AvgDistToGroundTruthAcc = size(1,iteration_num);
    AvgDistToGroundTruth = size(1,iteration_num);
    MinDistToGroundTruthAcc = zeros(4,iteration_num);
    MinDistToGroundTruth = zeros(4,iteration_num);
    AvgMetricResultAcc = zeros(3,iteration_num);
    AvgMetricResult = zeros(3,iteration_num);
    NumRawSolutions = size(1,iteration_num);
    
    for iteration = 1:iteration_num
        if num_obj == 1
            solutionMat = getSolutionsObjsAtOneIterationForSOGA(result, iteration); 
            % The reason of adding a negative is that fitness functions are minimized, so the real obj is -solutionMat.
            [pf, pfIdxs] = paretoFront(-solutionMat); 
            PFSolutions = getSolutionsFromPF(pfIdxs, result);
            
            solutionMatAcc = getSolutionsObjsForSOGA(result, iteration);
            [pfAcc, pfIdxsAcc] = paretoFront(-solutionMatAcc);
            PFSolutionsAcc = getSolutionsFromPF(pfIdxsAcc, result);            
        else
            solutionMat = getSolutionsObjsAtOneIteration(result, iteration);
            [pf, pfIdxs] = paretoFront(-solutionMat); % why we do a negative? fitness functions are minimized, so the real obj is -solutionMat.
            PFSolutions = getSolutionsFromPF(pfIdxs, result);
            
            % gets PF solutions at iteration K instead of first K iterations
            solutionMatAcc = getSolutionsObjs(result, iteration);
            [pfAcc, pfIdxsAcc] = paretoFront(-solutionMatAcc);
            PFSolutionsAcc = getSolutionsFromPF(pfIdxsAcc, result);
            
        end
        % # of solutions is size of PFSolutions.
        [WithInRangeSolutions, L2Dist_PFSolutions] = getWithInRangeSolutions(PFSolutions, ground_truth);
        [WithInRangeSolutionsAcc,~] = getWithInRangeSolutions(PFSolutionsAcc, ground_truth);
        [AvgDistToGroundTruth(iteration),MinDistToGroundTruth(:,iteration)] = getDistToGroundTruth(PFSolutions, ground_truth);
        [AvgDistToGroundTruthAcc(iteration),MinDistToGroundTruthAcc(:,iteration)] = getDistToGroundTruth(PFSolutionsAcc, ground_truth);
        if num_obj ~= 1
            AvgMetricResult(:,iteration) = getPFStats(pf);
            AvgMetricResultAcc(:,iteration) = getPFStats(pfAcc);
        end
        disp("iteration=" + iteration + " PFSolutionsSize="+size(PFSolutions,2)+" WithInRangeSolutionsSize="+size(WithInRangeSolutions,2) + " avg L2Dist_PFSolutions="+mean(L2Dist_PFSolutions));
        NumPFSolutionsAcc(iteration) = size(PFSolutionsAcc,2);
        NumWithInRangeSolutionsAcc(iteration) = size(WithInRangeSolutionsAcc,2);
        NumPFSolutions(iteration) = size(PFSolutions,2);
        NumWithInRangeSolutions(iteration) = size(WithInRangeSolutions,2);
        NumRawSolutions(iteration) = size(solutionMat,1);
    end
    disp("iteration_num=" + iteration_num + " PFSolutionsAccSize="+size(PFSolutionsAcc,2)+" WithInRangeSolutionsAccSize="+size(WithInRangeSolutionsAcc,2));
    
end

function [WithInRangeSolutions,L2Dist_PFSolutions] = getWithInRangeSolutions(PFSolutions, ground_truth)
    % TODO: check how many of PFSolutions are with-in range solutions
    % TODO: run K times, get min,max,avg, std dev
    global within_range_th;
    WithInRangeSolutions = [];
    L2Dist_PFSolutions = [];
    for col = 1:size(PFSolutions,2)
        dist = norm(ground_truth-PFSolutions(:,col).dec);
        L2Dist_PFSolutions = [L2Dist_PFSolutions, dist];
        if dist < within_range_th
            WithInRangeSolutions = [WithInRangeSolutions, PFSolutions(:,col)];
        end
    end
end

function [AvgMetricResult] = getPFStats(pf)
    % [avg metric1, avg metric2, avg metric3]'
    AvgMetricResult = mean(pf)';
end

function [AvgDist, MinDist] = getDistToGroundTruth(PFSolutions, ground_truth)
    L2Dist_PFSolutions = [];
    for col = 1:size(PFSolutions,2)
        dist = norm(ground_truth-PFSolutions(:,col).dec);
        L2Dist_PFSolutions = [L2Dist_PFSolutions, dist];
    end
    sortedDist = sort(L2Dist_PFSolutions);
    idxArray=[1,3,5,10];
    MinDist = [];
    last = realmax;
    for i=1:length(idxArray)
        endIdx = length(sortedDist);
        if idxArray(i)<length(sortedDist)
            endIdx = idxArray(i);
        end
        MinDist = [MinDist, mean(sortedDist(1:endIdx))];
    end
    AvgDist = mean(L2Dist_PFSolutions);
end

function [ p, idxs] = paretoFront( p )
% Filters a set of points P according to Pareto dominance, i.e., points
% that are dominated (both weakly and strongly) are filtered.
%
% Inputs: 
% - P    : N-by-D matrix, where N is the number of points and D is the 
%          number of elements (objectives) of each point.
%
% Outputs:
% - P    : Pareto-filtered P
% - idxs : indices of the non-dominated solutions
%
% Example:
% p = [1 1 1; 2 0 1; 2 -1 1; 1, 1, 0];
% [f, idxs] = paretoFront(p)
%     f = [1 1 1; 2 0 1]
%     idxs = [1; 2]
%
% Original Source: https://www.mathworks.com/matlabcentral/fileexchange/50477-pareto-filtering
[i, dim] = size(p);
idxs = [1 : i]';
while i >= 1
    old_size = size(p,1);
    indices = sum( bsxfun( @ge, p(i,:), p ), 2 ) == dim;
    indices(i) = false;
    p(indices,:) = [];
    idxs(indices) = [];
    i = i - 1 - (old_size - size(p,1)) + sum(indices(i:end));
end    
end

% get the objs of solutions of first k iterations from the result
function [solutions] = getSolutionsObjs(result, first_k_iterations)
    solutions =[];
    solu = result(:,2);
    for i = 1:first_k_iterations % assume first_k_iterations <= length(solu)
        if i>length(solu)
            break;
        end
        for j=1:length(solu{i,1})
            solutions = [solutions; solu{i,1}(j).obj];
        end        
    end
end

function [solutions] = getSolutionsObjsForSOGA(result, first_k_iterations)
    solutions =[];
    solu = result(:,2);
    for i = 1:first_k_iterations % assume first_k_iterations <= length(solu)
        if i>length(solu)
            break;
        end
        for j=1:length(solu{i,1})
            [metric,osMacMetric,osAndroidMetric,osWindowsMetric]=f_activeTimeOsMetricEqualWeightedSumSOGAWithBreakdown(solu{i,1}(j).dec);
            solutions = [solutions; [osMacMetric,osAndroidMetric,osWindowsMetric]];
        end        
    end
end

% TODO(jwu): use the k iteration also 
function [solutions] = getSolutionsObjsAtOneIteration(result, k_iterations)
    solutions =[];
    solu = result(:,2);
    i = k_iterations; % assume k_iterations <= length(solu)
    if i<=length(solu)        
        for j=1:length(solu{i,1})
            solutions = [solutions; solu{i,1}(j).obj];
        end
    end
end

function [solutions] = getSolutionsObjsAtOneIterationForSOGA(result, k_iterations)
    solutions =[];
    solu = result(:,2);
    i = k_iterations; % assume k_iterations <= length(solu)
    if i<=length(solu)        
        for j=1:length(solu{i,1})
            [metric,osMacMetric,osAndroidMetric,osWindowsMetric]=f_activeTimeOsMetricEqualWeightedSumSOGAWithBreakdown(solu{i,1}(j).dec);
            solutions = [solutions; [osMacMetric,osAndroidMetric,osWindowsMetric]];
        end
    end
end

% get the solutions of first k iterations from the result
function [solutions] = getSolutions(result, first_k_iterations)
    solutions =[];
    solu = result(:,2);
    for i = 1:first_k_iterations % assume first_k_iterations <= length(solu)
        if i>length(solu)
            break;
        end
        for j=1:length(solu{i,1})
            solutions = [solutions, solu{i,1}(j)];
        end        
    end
end

function [solutions] = getSolutionsAtOneIteration(result, k_iterations)
    solutions =[];
    solu = result(:,2);
    i = k_iterations; % assume k_iterations <= length(solu)
    if i<=length(solu)
        for j=1:length(solu{i,1})
            solutions = [solutions, solu{i,1}(j)];
        end 
    end
end


function Solutions = getSolutionsFromPF(pfIdxs,result)
	Solutions=[];
	solu = result(:,2);
    soluFlat = [];
    for j=1:length(solu)
        soluFlat = [soluFlat solu{j,1}];
    end
	for i=1:length(pfIdxs)
		Solutions=[Solutions, soluFlat(pfIdxs(i))];
	end
end

