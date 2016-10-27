a = EM;
a.q = 5;   %no of clusters

%class 1 training
class = 'faem0';
files = {'sa1' 'sa2' 'si1392' 'si2022' 'sx132' 'sx222'};
a = set_train(a,class,files);
a = K_means(a);
a = iterate(a,1);
a = set_prior_no(a,1);

%class 2 training
class = 'fdnc0';
files = {'sa1' 'sa2' 'si1278' 'si1908' 'sx18' 'sx288'};
a = set_train(a,class,files);
a = K_means(a);
a = iterate(a,2);
a = set_prior_no(a,2);

%class 3 training
class = 'fntb0';
files = {'sa1' 'sa2' 'si573' 'si679' 'sx123' 'sx213'};
a = set_train(a,class,files);
a = K_means(a);
a = iterate(a,3);
a = set_prior_no(a,3);

%class 4 training
class = 'mapv0';
files = {'sa1' 'sa2' 'si663' 'si1293' 'sx123' 'sx213'};
a = set_train(a,class,files);
a = K_means(a);
a = iterate(a,4);
a = set_prior_no(a,4);

%class 5 training
class = 'mdhs0';
files = {'sa1' 'sa2' 'si2160' 'sx180' 'sx360' 'sx450'};
a = set_train(a,class,files);
a = K_means(a);
a = iterate(a,5);
a = set_prior_no(a,5);

%class 6 training
class = 'mdlc0';
files = {'sa1' 'sa2' 'sx135' 'sx225' 'sx315' 'sx405'};
a = set_train(a,class,files);
a = K_means(a);
a = iterate(a,6);
a = set_prior_no(a,6);

%class 7 training
class = 'mjwt0';
files = {'sa1' 'sa2' 'si751' 'si1291' 'sx121' 'sx301'};
a = set_train(a,class,files);
a = K_means(a);
a = iterate(a,7);
a = set_prior_no(a,7);

%class 8 training
class = 'mlel0';
files = {'sa1' 'sa2' 'si1876' 'sx166' 'sx256' 'sx436'};
a = set_train(a,class,files);
a = K_means(a);
a = iterate(a,8);
a = set_prior_no(a,8);

%class 9 training
class = 'mrjb1';
files = {'sa1' 'sa2' 'si1020' 'si1413' 'sx30' 'sx300'};
a = set_train(a,class,files);
a = K_means(a);
a = iterate(a,9);
a = set_prior_no(a,9);

%class 10 training
class = 'msmc0';
files = {'sa1' 'sa2' 'si509' 'si1907' 'sx17' 'sx197'};
a = set_train(a,class,files);
a = K_means(a);
a = iterate(a,10);
a = set_prior_no(a,10);

a = set_prior(a);

%class 1 test
class = 'faem0';
files = {'sx312_10' 'sx312_11' 'sx312_12' 'sx312_13' 'sx312_14' 'sx402_20' 'sx402_21' 'sx402_22' 'sx402_23' 'sx402_24'};
a = get_conf(a,1,10,class,files);

%class 2 test
class = 'fdnc0';
files = {'sx108_10' 'sx108_11' 'sx108_12' 'sx108_13' 'sx108_14' 'sx198_20' 'sx198_21' 'sx198_22' 'sx198_23' 'sx198_24'};
a = get_conf(a,2,10,class,files);

%class 3 test
class = 'fntb0';
files = {'si1203_10' 'si1203_11' 'si1203_12' 'si1203_13' 'si1203_14' 'sx33_20' 'sx33_21' 'sx33_22' 'sx33_23' 'sx33_24'};
a = get_conf(a,3,10,class,files);

%class 4 test
class = 'mapv0';
files = {'si1923_20' 'si1923_21' 'si1923_22' 'si1923_23' 'si1923_24' 'sx33_10' 'sx33_11' 'sx33_12' 'sx33_13' 'sx33_14'};
a = get_conf(a,4,10,class,files);

%class 5 test
class = 'mdhs0';
files = {'sx90_20' 'sx90_21' 'sx90_22' 'sx90_23' 'sx90_24' 'sx270_10' 'sx270_11' 'sx270_12' 'sx270_13' 'sx270_14'};
a = get_conf(a,5,10,class,files);

%class 6 test
class = 'mdlc0';
files = {'si1395_10' 'si1395_11' 'si1395_12' 'si1395_13' 'si1395_14' 'sx45_20' 'sx45_21' 'sx45_22' 'sx45_23' 'sx45_24'};
a = get_conf(a,6,10,class,files);

%class 7 test
class = 'mjwt0';
files = {'si1381_10' 'si1381_11' 'si1381_12' 'si1381_13' 'si1381_14' 'sx211_20' 'sx211_21' 'sx211_22' 'sx211_23' 'sx211_24'};
a = get_conf(a,7,10,class,files);

%class 8 test
class = 'mlel0';
files = {'si1246_10' 'si1246_11' 'si1246_12' 'si1246_13' 'si1246_14' 'sx76_20' 'sx76_21' 'sx76_22' 'sx76_23' 'sx76_24'};
a = get_conf(a,8,10,class,files);

%class 9 test
class = 'mrjb1';
files = {'sx120_20' 'sx120_21' 'sx120_22' 'sx120_23' 'sx120_24' 'sx210_10' 'sx210_11' 'sx210_12' 'sx210_13' 'sx210_14'};
a = get_conf(a,9,10,class,files);

%class 10 test
class = 'msmc0';
files = {'si647_20' 'si647_21' 'si647_22' 'si647_23' 'si647_24' 'sx107_10' 'sx107_11' 'sx107_12' 'sx107_13' 'sx107_14'};
a = get_conf(a,10,10,class,files);

a.conf
