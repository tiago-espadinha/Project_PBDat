%% FEATURES - Load and show the images
load("girosmallveryslow2.mp4_features.mat");
features=double(features);
vid=VideoReader("girosmallveryslow2.mp4");
%% AULA 2 - Linear spaces
% Images from 5895:5906
seq1=[5895:5906];
seqfeat=features(:,5895:5906);
for i=1:length(seq1),
    mm{i}=vid.read(seq1(i));
end
montage(mm);%Just to see the images
%% ---compute the  stuff ! See notes and Zaki!!!!!
base=features(:,seq1);
[u s v]=svd(base,'econ');
Pi=u(:,1:12)*u(:,1:12)';
PiN=eye(512)-Pi;
fi=Pi*features;
fn=PiN*features;
di=sqrt(sum(fi .* fi));
dn=sqrt(sum(fn .* fn));
df=sqrt(sum(features .* features));
ri=di./df;
rnull=dn./df;
%indsi=find(ri >.75); - another way of defining inliers/outliers
%indsn=find(rnull>.85);
[~,indsi]=sort(ri,'descend');
[~,indsn]=sort(rnull,'descend');
%%
% SHOW THE IMAGES 
for i=1:100, mm{i}=vid.read(indsi(i));end
montage(mm,'Size',[10 10]);
pause;
for i=1:100, mm{i}=vid.read(indsn(i));end
montage(mm,'Size',[10 10]);
