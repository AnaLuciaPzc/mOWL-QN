function[x, iter, fun, psgrad, nonzero] = mOWL_online(X, Y, x0, maxiter, tol, m, lambda, epsi, b, c, lam, epsi1, tao, n0, impr, fileID )

%------------------------------------------------------------------------------------

[N, P] = size(X);

% imprimimos problema
fprintf(fileID,'------------------------------------------------------------------------- \n');
fprintf(fileID,'------------------------------------------------------------------------- \n');
fprintf(fileID,' \n');
fprintf(fileID,' \n');
fprintf(fileID,' Name of the problem: ...');
fprintf(fileID,' Number of variables:  % 5i  \n', P+1);
fprintf(fileID,' Number of observations:  % 5i  \n', N);
fprintf(fileID,'-------------------------------------- \n');

ET = Y;
X = [ones(N,1)  X];
X = sparse(X);
x = x0;

% Inicializamos variables.
%fev  = 1;
ind  = 1;
iter = 0;
%--------------------------------------------------------------------------
%
tic

% S Y para LBFGS
S = zeros(P+1,m);
Y = zeros(P+1,m);

% funcion y gradiente evaluados en el punto inical
batch = sort(randperm(N,b)); 
X1 = X(batch,:);
ET1 = ET(batch);
%f = feval('f_dev', X, ET, x);
f = feval('f_dev', X1, ET1, x);
g = feval('g_dev', X1, ET1,x);

% regularización con norma 1
f = f + lambda*norm(x,1);
fun = f;

% se obtiene el pseudo gradiente y su norma.
v = -Pseudo_g(g, x, lambda);

norm_v = norm(v,inf);
normv0 = norm_v;
psgrad = (norm_v/(1.0 + normv0))*ones(1,5); %Para poder checar el criterio de paro en las primeras iteraciones
nonzero= nnz(x);

fprintf(fileID,' Iter            f              ||g||     dif_f            alpha     s*y \n');
fprintf(fileID,' %3i  %21.15e   %8.2e  %8.5e   %5.3f    %8s  \n', ...
                            iter, f,  norm_v/(1.0 + normv0), [], [], []);
dif_f = Inf;

% Mientras no se cumpla la condición de paro o máximo de iteraciones.

iter1 = 4;
 
 while ( ( psgrad(iter1+1) > tol || psgrad(iter1) > tol  || psgrad(iter1-1) > tol  || psgrad(iter1-2) > tol   || psgrad(iter1-3) > tol   )  &&  iter < maxiter  )
    % linea 3 pseudo código (pc) mOWL
    v = -Pseudo_g(g, x, lambda); 
    
    % linea 4 pc mOWL
    idx = Indice(x,v,epsi);

   
    if iter >= 1
        
        % GD - step ( muy raro que se tome (else del código línea 14 )
        if idx ~= 0
            fprintf(fileID,'Se debió haber tomado el GD step.');
        end
        
        % QN - step línea  7-9 del código
        d = loop_LBFGS(S, Y, v, c, ind);
        p = Pi(d,v);
        
    else 
        
        % primera iteración:
        d = v;
        p = epsi1*d;
    end
    
    nt = (tao/(tao+iter))*n0;
    alpha = nt/c;
    
    xi  =  Xi(x, v); 
    xnew = Pi(x + alpha*p,xi);
    
    % 
    % ... almacenamos (s,y) 25-26 códgio
    s  = xnew - x;
    x  = xnew;
    f0 = f;
    g0 = g;
    
    % actualizamos
    f = feval('f_dev', X1, ET1, x);
    g = feval('g_dev', X1, ET1,x);
    % regularización con norma 1
    f = f + lambda*norm(x,1);
    fun = [fun f];
    dif_f = abs(f0 - f);
    y = g - g0 + lam*s;
    
    a = mod(iter,m) + 1;
    ind(a) = iter + 1;
    S(:,a) = s;
    Y(:,a) = y;
    sTy    = s'*y;
    
    if sTy <= eps
        fprintf(fileID,' Warning, small sTy < eps  % 8.2e \n', sTy);        
    end
   
    norm_v = norm(v,inf); 
    iter   = iter + 1;
    iter1 = iter1 + 1;
    
    psgrad = [psgrad norm_v/(1.0+normv0)];
    nonzero = [nonzero nnz(x)];
    
    if mod(iter,impr) == 0
       fprintf(fileID,' %3i  %21.15e   %8.2e   %8.5e   %5.3f    %8s  \n', iter, f, psgrad(iter1),dif_f, alpha, sTy );
    end
    
    batch = sort(randperm(N,b)); 
    length(batch);
    X1 = X(batch,:);
    ET1 = ET(batch);
    g = feval('g_dev', X1, ET1,x);
 end
toc

psgrad = psgrad(5:length(psgrad));
end

%-------------------------------------------------------------------
%-------------------------------------------------------------------


function [ f ] = f_dev( X, Y, w)

% ... objective function
%etiquetas en {0,1}
aux = X*w;
Log = log(1+exp(aux));
Log(aux>709) = aux(aux>709);
f = -mean( (Y.*aux) - Log );

end

function [ g ] = g_dev( X, Y, w)

[ b, p ] = size(X);

% ... objective function
%etiquetas en {0,1}
aux = X*w;
% ... Mini-batching gradient of the loss function
g = zeros(p,1);
term =   1./(1 + exp(aux));
g_aux = Y - (1-term);

g(1) = sum(g_aux); 
for i=2:p
    g(i) = sum( ( g_aux ).*X(:,i) );
end
g = -(1/b)*g;

end

%%-------------------------------------------------------------------------
%%-------------------------------------------------------------------------

function [flag] = Indice(x,v,eps)
% conjunto de índices que se usan en la línea 3 del pseudocódigo

    eps  = min(norm(v),eps);
    ind  = (abs(x)<=eps);
    ind2 = (x(ind).*v(ind) < 0);

    if sum(ind2)==0

        flag = 0;
    else
        flag = 1;
    end

end

%-------------------------------------------------------------------
%-------------------------------------------------------------------

function [r]= loop_LBFGS(s, y, g, c, indice)
    % Nocedal 197->
    % Alumno: Javier Urena Carrion
        % CU: 125319
    % Doble loop para encontrar producto H_k*grad_k en el metodo BFGS con
    % memoria limitada
    % Entrada:
        % s: matriz con vectores s
        % y: matriz con vectores y
        % g: gradiente en el punto actual
        % H0: aproximacion incial a H_k
        % indice: vector de indices que indique el orden de 's' e 'y' 
    % Salida:
        % r: Aproximacion a H_k * grad_k

    m = length(indice);
    [~,b]  = min(indice);
    [~,bm] = max(indice);
    q = g;
    a = b;
    ro = zeros(m,1);
    gam = zeros(m,1);
    for i = 1:m
       ro(a) = c / (y(:,a)' * s(:,a));
       gam(a) = 1 / (ro(a)*(y(:,a)' * y(:,a)));
       a = mod(a,m) + 1;
    end
    a = bm;
    alfa=zeros(m,1);
    for i=1:m
        alfa(a)= ro(a) * s(:,a)' * q;
        q = q - alfa(a) * y(:,a);
        a = mod(a - 2,m) + 1;
    end
    gamma = mean(gam);
    r = gamma * q;
    a = b;
    for i=1:m
        beta = ro(a) * y(:,a)' * r;
        r = r + (alfa(a) - beta) * s(:,a);
        a = mod(a,m) + 1;
    end
end

%-------------------------------------------------------------------
%-------------------------------------------------------------------

function [ z ] = Pi(x,y)
%Función Pi, como definida en el artículo.
n = length(x);
z = zeros(n,1);
a = (sign(x)==sign(y));

z(a>0) = x(a>0);
end

%-------------------------------------------------------------------
%-------------------------------------------------------------------

function [ p_g ] = Pseudo_g(g, x, lambda)
% función pseudo gradiente como definida en la sección 2.1
a     = (x>0);    b = (x<0);     c = (x==0);

p_g   = zeros(length(x),1);
aux = p_g; aux2= p_g;

p_g(a) = g(a)+lambda;
p_g(b) = g(b)-lambda;

aux(c) = (g(c)+lambda <0);
aux2(c) = (g(c)-lambda > 0); 
d = aux ==1;
e = aux2 ==1;

p_g(d)= g(d)+lambda;
p_g(e)= g(e)-lambda;
end

%-------------------------------------------------------------------
%-------------------------------------------------------------------

function [ xi ] = Xi( x,v )
% función xi del artículo
a = (x==0);
b = (x~=0);
xi(a) = sign(v(a));
xi(b) = sign(x(b));
xi    = xi';
end