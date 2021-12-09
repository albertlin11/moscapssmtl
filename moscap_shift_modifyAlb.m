clear; close all;
% constant
e0 = 8.854e-14;
eox = 3.9*e0;
esi = 11.9*e0;
k = 1.380e-20*1e-3;
q = 1.602e-19;
T = 300;
VT = k*T/q;
ni = 1.5e10;
% process parameters
na = 5.0e15;
tox = 200.0e-8;
phim = 4.28;
x = 4.05;
Eg = 1.12;
cox = eox/tox;
phib = VT*log(na/ni);
const = -0.78;
vfb = const+phim-x-Eg/2-phib;
Ld = sqrt(esi*VT/q/na);
%% 
% after deriving Qs¡APHIS as variable¡Aplot Qs vs PHIS
% Qs vs PHIS
PHIS = -0.4:0.0001:1.2;
Qs = sqrt(2*esi*k*T*na)*sqrt((exp(-PHIS/VT)+PHIS/VT-1)+ni*ni/na/na*(exp(PHIS/VT)-PHIS/VT-1));
figure;
plot(PHIS,abs(Qs),'LineWidth',2);
set(gcf,'position',[10, 10, 500, 500]);
set(gca,'yscale','log');
set(gca,'xscale','linear');
set(gca,'FontSize',12,'LineWidth',2,'FontWeight','bold');
xlabel('\psi_{s} (V)','FontName','Times New Roman','FontSize',12)
ylabel('|Q_{s}| (C/cm^{2})','FontName','Times New Roman','FontSize',12);
% title('|Qs| vs \psi');
axis([-0.4 1.1 1e-10 1e-4]);
% axis square;
% grid on;
% when PHIS<=0¡AQs is hole(positive)¡Aso Qssign is + ; when PHIS>0¡AQs is Na- or electron(negative)¡Aso Qssign is -
% +-Qs
for i = 1:length(Qs)
    if PHIS(i) <= 0
        Qssign(i) = Qs(i);
    end
    if PHIS(i) > 0
        Qssign(i) = -Qs(i);
    end
end
% differential Qs¡Aget the equation
% Cox in parallel¡Aget Ctotal
% change PHIS to V
for i = 1:length(PHIS)
    diffQs(i) = 0.5*sqrt(2*esi*k*T*na)*((-1/VT*exp(-PHIS(i)/VT)+1/VT)+ni*ni/na/na*(1/VT*exp(PHIS(i)/VT)-1/VT))*((exp(-PHIS(i)/VT)+PHIS(i)/VT-1)+ni*ni/na/na*(exp(PHIS(i)/VT)-PHIS(i)/VT-1))^-0.5;
end
csi = abs(diffQs);
for i = 1:length(csi)
    c(i) = (cox^-1+csi(i)^-1)^-1;
    if PHIS(i) == 0
        c(i) = (cox^-1+(esi/Ld)^-1)^-1;
    end
end
v = vfb-Qssign/cox+PHIS;
figure;
plot(v,c,'LineWidth',2);
set(gca,'FontSize',12,'LineWidth',2,'FontWeight','bold');
set(gcf,'position',[10, 10, 500, 500]);
xlabel('Voltage (V)','FontName','Times New Roman','FontSize',12);
ylabel('Capacitance per area (F/cm^{2})','FontName','Times New Roman','FontSize',12);
% title('CV curve in low frequency');
axis([-4.0 2.5 0.1e-7 1.8e-7]);
% axis square;
% grid on;
%% || V with same space
% deltaQ=C*deltaV
vv=-4.0:0.013:2.5;
cc = interp1(v,c,vv,'linear');
for i = 1:length(vv)
    if vv(i)>= const
        deltavv(i) = (vv(i)-vv(i-1));
        deltacc(i) = (cc(i)-cc(i-1));
        deltaQQ(i) = cc(i)*deltavv(i);
    end
end
% 2. find the voltage point of Qinv_th (using the approximate equation of inversion charge)
% Qs vs Qinv in different space v
PHIS = -0.4:0.0001:1.2;
Qs = sqrt(2*esi*k*T*na)*sqrt((exp(-PHIS/VT)+PHIS/VT-1)+ni*ni/na/na*(exp(PHIS/VT)-PHIS/VT-1));
Qinv = sqrt(2*esi*k*T*na)*sqrt(ni*ni/na/na*(exp(PHIS/VT)-PHIS/VT-1));
v = vfb-Qssign/cox+PHIS;
% Qs vs Qinv in same space v
vv=-4.0:0.013:2.5;
Qs = interp1(v,Qs,vv,'linear');
Qinv = interp1(v,Qinv,vv,'linear');
% deltaQs vs deltaQinv in same space
for i =1:length(vv)
    if vv(i) >= const
        deltaQs(i) = Qs(i)-Qs(i-1);
        deltaQinv(i) = Qinv(i)-Qinv(i-1);
    end
end
% get the maximum value
c=-log(0.0001)/1e-3;
Qinv1_th=max(deltaQinv)*(1-exp(-c*(1/1)*1e-3));
Qinv2_th=max(deltaQinv)*(1-exp(-c*(1/3)*1e-3));
Qinv3_th=max(deltaQinv)*(1-exp(-c*(1/5)*1e-3));
Qinv4_th=max(deltaQinv)*(1-exp(-c*(1/10)*1e-3));
Qinv5_th=max(deltaQinv)*(1-exp(-c*(1/50)*1e-3));
Qinv6_th=max(deltaQinv)*(1-exp(-c*(1/100)*1e-3));

deltaQinv1 = deltaQinv;
deltaQinv2 = deltaQinv;
deltaQinv3 = deltaQinv;
deltaQinv4 = deltaQinv;
deltaQinv5 = deltaQinv;
deltaQinv6 = deltaQinv;
for i =1:length(vv)
    if vv(i) >= const
        if deltaQinv1(i) >= Qinv1_th
            deltaQinv1(i) = Qinv1_th;
        end
        if deltaQinv2(i) >= Qinv2_th
            deltaQinv2(i) = Qinv2_th;
        end
        if deltaQinv3(i) >= Qinv3_th
            deltaQinv3(i) = Qinv3_th;
        end
        if deltaQinv4(i) >= Qinv4_th
            deltaQinv4(i) = Qinv4_th;
        end
        if deltaQinv5(i) >= Qinv5_th
            deltaQinv5(i) = Qinv5_th;
        end
        if deltaQinv6(i) >= Qinv6_th
            deltaQinv6(i) = Qinv6_th;
        end
    end
end
figure;
plot(vv,deltaQinv1,vv,deltaQinv2,vv,deltaQinv3,vv,deltaQinv4,vv,deltaQinv5,vv,deltaQinv6,'LineWidth',2);
set(gca,'FontSize',12,'LineWidth',2,'FontWeight','bold');
set(gcf,'position',[10, 10, 600, 500]);
legend('Location','northeastoutside');
legend('1k','3k','5k','10k','50k','100k');
xlabel('Voltage (V)','FontName','Times New Roman','FontSize',12);
ylabel('\DeltaQ_{inv} (C/cm^{2})','FontName','Times New Roman','FontSize',12);
% title('deltaQinv vs voltage');
xlim([const 2.5]);
% grid on;
% get the vth of each frequency
for i = 1:length(vv)
    if vv(i) >= const && deltaQinv1(i) > Qinv1_th
       break
    end
end
v1_th = vv(i);
for i = 1:length(vv)
    if vv(i) >= const && deltaQinv2(i) == Qinv2_th
       break
    end
end
v2_th = vv(i);
for i = 1:length(vv)
    if vv(i) >= const && deltaQinv3(i) == Qinv3_th
       break
    end
end
v3_th = vv(i);
for i = 1:length(vv)
    if vv(i) >= const && deltaQinv4(i) == Qinv4_th
       break
    end
end
v4_th = vv(i);
for i = 1:length(vv)
    if vv(i) >= const && deltaQinv5(i) == Qinv5_th
       break
    end
end
v5_th = vv(i);
for i = 1:length(vv)
    if vv(i) >= const && deltaQinv6(i) == Qinv6_th
       break
    end
end
v6_th = vv(i);
% plot cv curve
cc1 = cc;
cc2 = cc;
cc3 = cc;
cc4 = cc;
cc5 = cc;
cc6 = cc;
for i = 1:length(vv)
    if vv(i) > v1_th
        cc1(i) = cc1(i-1);
    end
    if vv(i) > v2_th
        cc2(i) = cc2(i-1);
    end
    if vv(i) > v3_th
        cc3(i) = cc3(i-1);
    end
    if vv(i) > v4_th
        cc4(i) = cc4(i-1);
    end
    if vv(i) > v5_th
        cc5(i) = cc5(i-1);
    end
    if vv(i) > v6_th
        cc6(i) = cc6(i-1);
    end
end
figure;
plot(vv,cc1,vv,cc2,vv,cc3,vv,cc4,vv,cc5,vv,cc6,'LineWidth',2);
set(gca,'FontSize',12,'LineWidth',2,'FontWeight','bold');
set(gcf,'position',[10, 10, 600, 500]);
legend('Location','northeastoutside');
legend('1k','3k','5k','10k','50k','100k');
xlabel('Voltage (V)','FontName','Times New Roman','FontSize',12);
ylabel('Capacitance per area (F/cm^{2})','FontName','Times New Roman','FontSize',12);
% title('CV curve');
xlim([-4.0 2.5]);
% grid on;
% Output excel file
% vv = reshape(vv,501,1);
% cc1 = reshape(cc1,501,1);
% cc2 = reshape(cc2,501,1);
% cc3 = reshape(cc3,501,1);
% cc4 = reshape(cc4,501,1);
% cc5 = reshape(cc5,501,1);
% cc6 = reshape(cc6,501,1);
% B=[vv,cc1,cc2,cc3,cc4,cc5,cc6];
% xlswrite('theoretical.xlsx', cellstr('Voltage'),'A1:A1');
% xlswrite('theoretical.xlsx', cellstr('1k'),'B1:B1');
% xlswrite('theoretical.xlsx', cellstr('3k'),'C1:C1');
% xlswrite('theoretical.xlsx', cellstr('5k'),'D1:D1');
% xlswrite('theoretical.xlsx', cellstr('10k'),'E1:E1');
% xlswrite('theoretical.xlsx', cellstr('50k'),'F1:F1');
% xlswrite('theoretical.xlsx', cellstr('100k'),'G1:G1');
% xlswrite('theoretical.xlsx', B,'A2:G502');