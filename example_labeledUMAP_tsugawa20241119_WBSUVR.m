%%%% UMAP
% ver.1 2023.9.4
% author:noguchi hinako

% 保存先ディレクトリを指定
clear;

currentDate = datetime('now', 'Format', 'yyyyMMdd');

% ディレクトリ名を作成
saveDir = fullfile('./result/', char(currentDate),'/5umap');

M = readtable("./data/HCPSY219_WBSUVR_Dataset.csv");


%manual WB
WB_ASD = M(1:35,8:82);
WB_BIP = M(36:72,8:82);
WB_DEP = M(73:107,8:82);
WB_HC = M(108:177,8:82);
WB_SCH = M(178:219,8:82);


%rmmissing
rmWB_ASD = rmmissing(WB_ASD);
rmWB_BIP = rmmissing(WB_BIP);
rmWB_DEP = rmmissing(WB_DEP);
rmWB_HC = rmmissing(WB_HC);
rmWB_SCH = rmmissing(WB_SCH);
%欠損値補完
WB_ASD = fillmissing(WB_ASD, 'nearest');
WB_BIP = fillmissing(WB_BIP, 'nearest');
WB_DEP = fillmissing(WB_DEP, 'nearest');
WB_HC  = fillmissing(WB_HC, 'nearest');
WB_SCH = fillmissing(WB_SCH, 'nearest');

% テーブルを縦に結合する(全部がNAなDEP２症例は削除された状態)
exactD_table = vertcat(WB_ASD, WB_BIP, WB_DEP, WB_HC, WB_SCH);

% テーブルを配列へ変換
exactD = table2array(exactD_table);

% ROI毎に被験者方向で正規化
exactD = normalize(exactD,1);

%give labels
label_ASD = ones(size(WB_ASD,1),1)*UDisease.ASD;
label_BIP = ones(size(WB_BIP,1),1)*UDisease.bipolar;
label_DEP = ones(size(WB_DEP,1),1)*UDisease.depression;
label_HC = ones(size(WB_HC,1),1)*UDisease.HC;
label_SCH = ones(size(WB_SCH,1),1)*UDisease.schizophrenia;

labels = vertcat(label_ASD,label_BIP,label_DEP,label_HC,label_SCH);

labeled_exactD = horzcat(exactD,labels);


% UMAPのパラメータを設定する
%rng('default'); % 乱数
%n_neighbors = 10; % 近傍点の数
%min_dist = 0.1; % 最小距離
%n_components = 2; % 次元数

% UMAPのパラメータを設定する
rng('default'); % 乱数
n_neighbors = 50; % 近傍点の数
min_dist = 1; % 最小距離
n_components = 2; % 次元数
spread = 10;

% run_umap関数を実行する
[reduction, umap, clusterIds, extras] = ...
    run_umap(labeled_exactD,...
    'label_column','end',...
    'target_weight',0.25,...
    'n_neighbors', n_neighbors,...
    'spread', spread,...
    'min_dist', min_dist,...
    'n_components', n_components,...
    'randomize', false,...
    'marker_size',5,...
    'marker','o',...
    'contour_percent',0);

asd = reduction(1:35,:);
bip = reduction(36:72,:);
dep = reduction(73:107,:);
hc = reduction(108:177,:);
sch = reduction(178:219,:);

%%%%%%%%%%%%%%%%%%%%%%%%% 一枚のウィンドウで表示 %%%%%%%%%%%%%%%%%%%%%%%%%%%
fig = figure;

% モニタの最大画面サイズに設定
set(fig, 'Position', get(0, 'Screensize'));

% 新しいウィンドウを作成してサブプロットを設定
tiledlayout(2,3)

% 保存先ディレクトリが存在しない場合は作成
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%  全部を表示  %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% light_grayを定義
light_gray = [100, 100, 100] / 255; % RGB値を0から1の範囲に正規化

nexttile;
% プロットの作成
scatter(asd(:,1), asd(:,2), 20, 'MarkerFaceColor', '#FFA500', 'MarkerEdgeColor', '#FFA500', 'Marker', 'square');
hold on
scatter(bip(:,1), bip(:,2), 25, 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'red', 'Marker', 'pentagram');
scatter(dep(:,1), dep(:,2), 30, 'MarkerFaceColor', 'blue', 'MarkerEdgeColor', 'blue', 'Marker', 'hexagram');
scatter(hc(:,1), hc(:,2), 20, 'MarkerFaceColor', 'yellow', 'MarkerEdgeColor', 'yellow', 'Marker', 'o');
scatter(sch(:,1), sch(:,2), 20, 'MarkerFaceColor', 'green', 'MarkerEdgeColor', 'green', 'Marker', 'diamond');

ax = gca;
ax.XTickLabel = "";
ax.YTickLabel = "";
% 凡例の作成
% マーカーの表記を追加
legend('ASD','BIP','DEP', 'HC', 'SCH','Location', 'best');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%   ASD   %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nexttile;
% プロットの作成
others = cat(1,bip,dep,hc,sch);

% ASD プロット
scatter(asd(:,1), asd(:,2), 20, 's', 'MarkerFaceColor', '#FFA500', 'MarkerEdgeColor', '#FFA500', 'DisplayName', 'ASD');
hold on
% Others プロット
scatter(others(:,1), others(:,2), 20, 'o', 'MarkerFaceColor', light_gray, 'MarkerEdgeColor', light_gray, 'DisplayName', 'Others');

ax = gca;
ax.XTickLabel = "";
ax.YTickLabel = "";
% 凡例の作成
legend('Location', 'best');  % 凡例を自動配置

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%   BIP   %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nexttile;
% プロットの作成
others = cat(1,asd,dep,hc,sch);
% BIP プロット
scatter(bip(:,1), bip(:,2), 25, 'p', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'red', 'DisplayName', 'BIP');
hold on
% Others プロット
scatter(others(:,1), others(:,2), 20, 'o', 'MarkerFaceColor', light_gray, 'MarkerEdgeColor', light_gray, 'DisplayName', 'Others');
ax = gca;
ax.XTickLabel = "";
ax.YTickLabel = "";
% 凡例の作成
legend('Location', 'best');  % 凡例を自動配置


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%   DEP   %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nexttile;
% プロットの作成
others = cat(1,asd,bip,hc,sch);
% DEP プロット
scatter(dep(:,1), dep(:,2), 30, 'h', 'MarkerFaceColor', 'blue', 'MarkerEdgeColor', 'blue', 'DisplayName', 'DEP');
hold on
% Others プロット
scatter(others(:,1), others(:,2), 20, 'o', 'MarkerFaceColor', light_gray, 'MarkerEdgeColor', light_gray, 'DisplayName', 'Others');
ax = gca;
ax.XTickLabel = "";
ax.YTickLabel = "";
% 凡例の作成
legend('Location', 'best');  % 凡例を自動配置


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%   HC    %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nexttile;
% プロットの作成
others = cat(1,asd,bip,dep,sch);
% HC プロット
scatter(hc(:,1), hc(:,2), 20, 's', 'MarkerFaceColor', 'yellow', 'MarkerEdgeColor', 'yellow', 'DisplayName', 'HC');
hold on
% Others プロット
scatter(others(:,1), others(:,2), 20, 'o', 'MarkerFaceColor', light_gray, 'MarkerEdgeColor', light_gray, 'DisplayName', 'Others');
ax = gca;
ax.XTickLabel = "";
ax.YTickLabel = "";

% 凡例の作成
legend('Location', 'best');  % 凡例を自動配置




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%   SCH   %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nexttile;
% プロットの作成
others = cat(1,asd,bip,dep,hc);
% SCH プロット
scatter(sch(:,1), sch(:,2), 20, 'd', 'MarkerFaceColor', 'green', 'MarkerEdgeColor', 'green', 'DisplayName', 'SCH');
hold on
% Others プロット
scatter(others(:,1), others(:,2), 20, 'o', 'MarkerFaceColor', light_gray, 'MarkerEdgeColor', light_gray, 'DisplayName', 'Others');
ax = gca;
ax.XTickLabel = "";
ax.YTickLabel = "";
% 凡例の作成
legend('Location', 'best');  % 凡例を自動配置


% ファイル名を指定してPNGとして保存（パスを指定）
saveas(fig, fullfile(saveDir, 'ラベルありUMAP_WB219_20241119.png'));

close all;