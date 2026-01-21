%% PRELIMINARIES
%==========================================================================
clear all; clear session; close all; clc
warning off all

% Load data
[aux_xlsdata, aux_xlstext] = xlsread('data/data_estimation.xlsx','data');

%%
% Model
xlsdata=aux_xlsdata;
xlstext=aux_xlstext;

%%
X = xlsdata;
dates = xlstext(2:end,1);
vnames_long = xlstext(1,2:end);
vnames = xlstext(1,2:end);
nvar = length(vnames);
data   = Num2NaN(xlsdata);

% Store variables in the structure DATA
for ii=1:length(vnames)
    DATA.(vnames{ii}) = data(:,ii);
end
% Convert the first date to numeric
year = str2double(xlstext{2,1}(1:4));
quarter = str2double(xlstext{2,1}(6));
% Observations
nobs = size(data,1);

%% VAR ESTIMATION
%==========================================================================
% Set endogenous

% Model
VARvnames_long = {'rstar_up'  'debtgdp'  'infla'	'UR'	'FFR'	 ;};
VARvnames      = {'rstar_up'  'debtgdp'   'infla'	'UR'	'FFR'	 ;};
VARnvar        = length(VARvnames);


%%

% Set deterministics for the VAR
 VARconst = 1;
% Set number of nlags
VARnlags = 4;

% Create matrices of variables for the VAR
ENDO = nan(nobs,VARnvar);
for ii=1:VARnvar
    ENDO(:,ii) = DATA.(VARvnames{ii});
end
%ENDO=ENDO(1:164,:); % cuando uso TAX shock
% si corto la muestra para que se tenga el mismo tamaño que 

%%
% Estimate VAR
[VAR, VARopt] = VARmodel(ENDO,VARnlags,VARconst);

% Print estimation on screen
%VARopt.vnames = vnames;
%[TABLE, beta] = VARprint(VAR,VARopt,2);

%% COMPUTE IR AND VD
%==========================================================================
% Set options some options for IRF calculation
VARopt.nsteps = 12;
VARopt.ident = 'short';
VARopt.vnames = VARvnames_long;
VARopt.FigSize = [26,12];
VARopt.impact = 1;
VARopt.ndraws = 10000;
VARopt.pctg   = 90;  % change for 68 pctg
% Compute IRF
[IRF, VAR] = VARir(VAR,VARopt);
%VARirplot(IRF,VARopt)
% Compute error bands
[IRinf,IRsup,IRmed,IRbar] = VARirband(VAR,VARopt);
% Plot
%VARirplot(IRbar,VARopt,IRinf,IRsup);


%% Plot for real rate
figure(1)
FigSize(20,20)
plot(IRbar(:,1,2),'--r','LineWidth',1); hold on; 
plot(IRinf(:,1,2),'--r','LineWidth',1); hold on; 
plot(IRsup(:,1,2),'--r','LineWidth',1); hold on; 
plot(zeros(VARopt.nsteps),'-k')
title(VARvnames_long{1},'FontWeight','bold')

%% Plot for debt to gdp
figure(2)
FigSize(20,20)
plot(IRbar(:,2,2),'--r','LineWidth',1); hold on; 
plot(IRinf(:,2,2),'--r','LineWidth',1); hold on; 
plot(IRsup(:,2,2),'--r','LineWidth',1); hold on; 
plot(zeros(VARopt.nsteps),'-k')
title(VARvnames_long{2},'FontWeight','bold')

return
%IRFinf90=IRinf(:,1,2);
%IRsup90=IRsup(:,1,2);
IRmean=IRbar(:,1,2);

IRinf68=IRinf(:,1,2);
IRsup68=IRsup(:,1,2);
