s_x = 'PATH_TO_SOURCE_EEG';
s_y = 'PATH_TO_SOURCE_LABEL';
t_x = 'PATH_TO_TARGET_EEG';
t_y = 'PATH_TO_TARGET_LABEL';

fs = 100;
fLow = 8;
fHigh = 32;
fOrder = 8;
p = 20;
maxEpoch = 5;

nS = size(s_x, 3);
nTtr = 150;

s_x = permute(filterFun(permute(s_x, [2,1,3]), fs, fLow, fHigh, fOrder), [2,1,3]);
t_x = permute(filterFun(permute(t_x, [2,1,3]), fs, fLow, fHigh, fOrder), [2,1,3]);

t_y_train = t_y(1:nTtr);
t_y_test = t_y(nTtr+1:end);

s_Covs = covFun(s_x);
t_Covs = covFun(t_x);

nCh = size(s_Covs, 1);
tanDim = nCh * (nCh + 1) / 2;

s_Cov = squeeze(num2cell(s_Covs, [1 2]))';
t_train_Cov = squeeze(num2cell(t_Covs(:,:,1:nTtr), [1 2]))';
t_test_Cov = squeeze(num2cell(t_Covs(:,:,nTtr+1:end), [1 2]))';

acc_train = zeros(maxEpoch, 1);
acc_test = zeros(maxEpoch, 1);

mPlan = SinkhornRegOptimalTransport(s_Cov, t_train_Cov, s_y);
s_CovOT = ApplyPlan(s_Cov, t_train_Cov, mPlan, p);

rm = RiemannianMean(cat(3, s_CovOT{:}, t_train_Cov{:}));
mX = CovsToVecs(cat(3, s_CovOT{:}, t_train_Cov{:}, t_test_Cov{:}), rm);

[pre_train, pre_test] = svmPredict(mX, s_y, nS, nTtr);
acc_train(1) = mean(pre_train == t_y_train');
acc_test(1) = mean(pre_test == t_y_test');

for epoch = 2:maxEpoch
    Covs = cat(3, s_Cov{:}, t_train_Cov{:});
    MCov = RiemannianMean(Covs);

    TVecs = zeros(tanDim, size(Covs, 3));
    for j = 1:size(Covs, 3)
        TVecs(:,j) = CovToTan(Covs(:,:,j), MCov);
    end

    s_vecs = TVecs(:, 1:nS);
    t_vecs = TVecs(:, nS+1:nS+nTtr);

    s0_mean = mean(s_vecs(:, s_y == 0), 2);
    s1_mean = mean(s_vecs(:, s_y == 1), 2);
    t0_mean = mean(t_vecs(:, pre_train == 0), 2);
    t1_mean = mean(t_vecs(:, pre_train == 1), 2);

    [~, new_s1_vecs] = centroid_alignment(s0_mean, s1_mean, t0_mean, t1_mean, s_vecs(:, s_y == 1));

    new_s_vecs = s_vecs;
    new_s_vecs(:, s_y == 1) = new_s1_vecs;

    new_s_Cov = cell(1, nS);
    for j = 1:nS
        new_s_Cov{j} = TanToCov(new_s_vecs(:,j), MCov);
    end

    l_s_Cov = new_s_Cov(s_y == 0);
    l_t_Cov = t_train_Cov(pre_train' == 0);
    l_mPlan = SinkhornRegOptimalTransport(l_s_Cov, l_t_Cov, s_y(s_y == 0));
    l_CovOT = ApplyPlan(l_s_Cov, l_t_Cov, l_mPlan, p);

    r_s_Cov = new_s_Cov(s_y == 1);
    r_t_Cov = t_train_Cov(pre_train' == 1);
    r_mPlan = SinkhornRegOptimalTransport(r_s_Cov, r_t_Cov, s_y(s_y == 1));
    r_CovOT = ApplyPlan(r_s_Cov, r_t_Cov, r_mPlan, p);

    s_CovOT = cell(1, nS);
    s_CovOT(s_y == 0) = l_CovOT;
    s_CovOT(s_y == 1) = r_CovOT;

    rm = RiemannianMean(cat(3, s_CovOT{:}, t_train_Cov{:}));
    mX = CovsToVecs(cat(3, s_CovOT{:}, t_train_Cov{:}, t_test_Cov{:}), rm);

    [pre_train, pre_test] = svmPredict(mX, s_y, nS, nTtr);
    acc_train(epoch) = mean(pre_train == t_y_train');
    acc_test(epoch) = mean(pre_test == t_y_test');
end

function [pre_train, pre_test] = svmPredict(mX, s_y, nS, nTtr)
    mX_s = mX(:, 1:nS);
    mX_t_train = mX(:, nS+1:nS+nTtr);
    mX_t_test = mX(:, nS+nTtr+1:end);

    [idx, Fs] = Fvalue(mX_s', s_y, length(s_y));
    mX_s = Fs';
    mX_t_train = mX_t_train(idx, :);
    mX_t_test = mX_t_test(idx, :);

    mdl = fitcecoc(mX_s', s_y, 'Learners', templateSVM('Standardize', false));
    pre_train = mdl.predict(mX_t_train');
    pre_test = mdl.predict(mX_t_test');
end

function Covs = covFun(X)
    Covs = zeros(size(X, 1), size(X, 1), size(X, 3));
    for i = 1:size(X, 3)
        E = X(:,:,i);
        C = E * E';
        Covs(:,:,i) = C / trace(C);
    end
end

function [adjusted_svec, new_vecs] = centroid_alignment(s0, s1, t0, t1, svecs)
    svec = s1 - s0;
    tvec = t1 - t0;

    dir_t = tvec / norm(tvec);
    adjusted_svec = dir_t * norm(svec);

    new_s1 = s0 + adjusted_svec;
    tran = new_s1 - s1;

    new_vecs = zeros(size(svecs));
    for i = 1:size(svecs, 2)
        new_vecs(:,i) = svecs(:,i) + tran;
    end
end
