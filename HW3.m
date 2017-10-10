load fisheriris.mat

y = zeros(length(species),1);
for i = 1:length(species)
  if strcmp(species{i,1},'setosa')
    s = 1;
  elseif strcmp(species{i,1},'versicolor')
    s = 2;
  else
    s = 3;
    
  end
  y(i) = s;
end

iris = [meas, y];

for bins = 5:5:20
  [n, x] = hist(iris(:,1),bins);
  
  % how to compute entropy 
  % P = [p1, ..., pn] is prob distribution
  % log2P is a vector of log2 values: log2p1, ..., log2pn
  % p1 x log2 p1 + ... + pn log2 pn is the dot product of these 
  % two vectors
  % example:
  n

  % form p:
  p = n/sum(n)

  %check that it is a prob
  sum(p)

  % form logs
  logp = log2(p)

  % Compute the entropy: - dot product
  Entropy= -logp*p'  or Entropy = -sum(logp .* p)

  
end