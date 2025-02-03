%% F0 + F0' + F0'' matrices
function [m, mp, m2p, times] = F0_matrix(Q, PI_, G, rho)
    tic;
    logG = log(G);
    G1 = exp(-rho/(1 + rho)*logG);
    G2 = exp(1/(1 + rho)*logG);
    qg2 = Q.'*G2;
    logqg2 = log(qg2);
    qg2rho = exp(rho*logqg2);
    pig1 = PI_.*G1;
    initial = toc;

    tic;
    m = Q.'*pig1*(qg2rho.');
    m = (1/pi)*m;
    first = toc;

    tic;
    mp = Q.'*pig1*((qg2rho.*logqg2).') - 1/(1 + rho) * Q.'*(pig1.*logG)*(qg2rho.');
    mp = (1/pi)*mp;
    second = toc;

    tic
    m2p = Q.'*pig1*((qg2rho.*logqg2.*logqg2).') - 2/(1 + rho) * Q.'*(pig1.*logG) * ((qg2rho.*logqg2).') + 1/(1 + rho)^2 * Q.'*(pig1.*logG.*logG)*(qg2rho.');
    m2p = (1/pi)*m2p;
    third = toc;

    times = [initial, first, second, third];

end
