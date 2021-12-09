clc;
close all;
model = {'MLP','SSMTL','TL',};
neural = {'50','50','5050'};       % 50 neurons plot data  
%%%%%%%%%%%%%%%%%%%%% C-V PLOT on Train data %%%%%%%%%%%%%%%%%%%%%
Train = cell(1,3);
for j = 1:length(model)
    %Train{j} = readtable(['.\' model{j} '\csv\'  model{j} '_train_60.csv']); 
    Train{j} = readtable(['.\' model{j} '\csv\'  model{j} '_train_' neural{j} '.csv']); 
    %Train{j} = readtable(['.\'  model{j} '_train_' neural{j} '.csv']); 
    
    figure('units','centimeter','position',[2, 2, 10, 7.5])
    h = plot(Train{1,j}{1:51,6},Train{1,j}{1:51,7},Train{1,j}{1:51,6},Train{1,j}{1:51,8},Train{1,j}{1021:1071,6},Train{1,j}{1021:1071,7},Train{1,j}{1021:1071,6},Train{1,j}{1021:1071,8});
    NameArray = {'Marker','Color'};
    ValueArray = {'o','blue';'^','black';'*','red';'o','cyan'};
    set(h,NameArray,ValueArray)
    set(gca,'FontSize', 12, 'LineWidth', 2)
    xlim([-4 3]), ylim([1 9])%, xticks([0:0.05:0.22]), yticks([0:20:100])
    xlabel('\bfVoltage (V)', 'FontSize',12), ylabel('\bfCapacitance(pF)', 'FontSize',12)
    legend({'\bf LF:Measured','\bf LF:Predict','\bf HF:Measured','\bf HF:Predict'},'FontSize',10, 'Location','southeast','Box','off')
end
%%%%%%%%%%%%%%%%%%%%% Scatter PLOT for  TEST %%%%%%%%%%%%%%%%%%%%%
% i=[ 1- MLP, 2-SSMTL, 3-TL ]
Test = cell(1,3);
data = cell(3,8);
for i = 1:length(model)
    %Test{i} = readtable(['.\' model{i} '\csv\'  model{i} '_test_60.csv']);    
    Test{i} = readtable(['.\' model{i} '\csv\'  model{i} '_test_' neural{i} '.csv']); 
    
    freq = [3 10 50];
    for j = 1:length(freq)            %% j=[ 1=3kHz,  2=10kHz, 3=50kHz,]
        %freq=[3 5 10 50]
        data{i,j} = Test{1,i}{any(Test{1,i}{:,:}==freq(j),2),:};
        figure('units','centimeter','position',[2, 2, 10, 7.5])
        plot([-5 25],[-5 25],'--','LineWidth', 2,'color','k')
        %xlim([0, 23]), ylim([0, 23]),xticks([0:5:25]), yticks([0:5:25])
        hold on
        h = plot(data{i,j}(:,7),data{i,j}(:,8),'o',   'Markersize', 4, 'MarkerEdgeColor', 'b', 'MarkerFaceColor','b');
        R = R2(data{i,j}(:,7),data{i,j}(:,8));
        R=round(R,4);  
        txt = ['\bfR^2: ' num2str(R)];
        text(15,3.5,txt);
        set(gca,'FontSize', 12, 'LineWidth', 2);
        xlim([0 23]), ylim([0 23]),xticks(0:5:25), yticks(0:5:25),
        xlabel('\bfMeasured capacitance (pF)', 'FontSize',12), ylabel('\bfPredicted capacitance (pF)', 'FontSize',12);
        legend({'\bfIdeal','\bfPredict'},'FontSize',12, 'Location','northwest','Box','off');
    end
end
%%%%%%%%%%%%%%%%%%%%% Intelligent Manufacturing %%%%%%%%%%%%%%%%%%%%%
% Condition: Both Dry and Wet......Frequency=50kHz, Area=0.01, Clean Type=RCA, Metal Dep= E-gun
for i = 1:3
    for j = 1:8    % Due to 8 Column in dataset
        data{i,j} = Test{1,i}{:,j};
    end
    keyIndex = (data{i,1}==50) &(data{i,2}==0.01) &(data{i,3})==0 &(data{i,5}==1);
    model{i} = [data{i,1}(keyIndex) data{i,2}(keyIndex) data{i,3}(keyIndex) data{i,4}(keyIndex) data{i,5}(keyIndex) data{i,6}(keyIndex) data{i,7}(keyIndex) data{i,8}(keyIndex) ];
    figure('units','centimeter','position',[2, 2, 10, 7.5])
    t = plot( model{i}(1:51,6), model{i}(1:51,7),model{i}(52:102,6), model{i}(52:102,7),model{i}(1:51,6), model{i}(1:51,8), model{i}(52:102,6), model{i}(52:102,8));
    NameArray = {'Marker','Color'};
    %ValueArray = {'-','blue';'^','black';'-','red';'^','cyan'};
    ValueArray = {'o','blue';'^','black';'*','red';'o','cyan'};
    set(t,NameArray,ValueArray)
    set(gca,'FontSize', 12, 'LineWidth', 2)
    xlabel('\bfVoltage (V)', 'FontSize',12), ylabel('\bfCapacitance(pF)', 'FontSize',12);
    legend({'\bf Wet:Measured','\bf Dry:Measured','\bf Wet:Predict','\bf Dry:Predict'},'FontSize',10, 'Location','northeast','Box','off');
end