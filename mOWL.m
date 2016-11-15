function[x, iter, fun, psgrad, nonzero] = mOWL(X, Y, x0, maxiter, tol, m, lambda, epsi, alpha, beta, gam, impr )
% Implementación del algoritmo modificado OWL que se describe en el
% artículo "A Modified Orhant-Wise Limited Memory Quasi-Newton Method with
% Convergence Analysis" de Gong y Ye.
%                                           Ana Lucía Pérez
%                                           ITAM, 2016  
%
%    *Input:
% Las variables de entrada son como se decriben en el artículo mencionado,
% con X la matriz de datos, Y el vector de etiquetas, x0 el punto inicial,
% impr cada cuántas iteraciones se está imprimiendo la salida.
%    *Output:
% x el punto óptimo, iter el número de iteraciones que se realizaron, fun
% el valor de la función objetivo en cada iteración. 
%%------------------------------------------------------------------------------------

[N, p] = size(X);
% imprimimos problema
disp(' Name of the problem: ...');
fprintf(' Number of variables:  % 5i  \n', p+1);
fprintf(' Number of observations:  % 5i  \n', N);
fprintf('-------------------------------------- \n');

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
S = zeros(p+1,m);
Y = zeros(p+1,m);

% funcion y gradiente evaluados en el punto inical
[ f, g ] = feval( 'FunDer', X, ET, x);

% regularización con norma 1
f = f + lambda*norm(x,1);
fun = f;

% se obtiene el pseudo gradiente y su norma.
v = -Pseudo_g(g, x, lambda);

norm_v = norm(v,inf);
normv0 = norm_v;
psgrad = norm_v/(1.0 + normv0);
nonzero= nnz(x);

fprintf(1,' Iter            f              ||g||     dif_f            alpha     s*y \n');
fprintf(' %3i  %16.10e   %8.2e  %8.5e   %5.3f    %5s  \n', ...
                            iter, f,  norm_v/(1.0 + normv0), [], [], []);

maxrec = 10;

dif_f = Inf;

% Mientras no se cumpla la condición de paro o máximo de iteraciones.

while  dif_f > tol  &&  iter < maxiter
    % linea 3 pseudo código (pc) mOWL
    v = -Pseudo_g(g, x, lambda); 
    
    % linea 4 pc mOWL
    idx = Indice(x,v,epsi);

   
    if iter >= 1
        
        % GD - step ( muy raro que se tome (else del código línea 14 )
        if idx ~= 0
            warning('se debió haber tomado el GD step.');
        end
        
        % QN - step línea  7-9 del código
        d = loop_LBFGS(S, Y, v, gamma_k, ind);
        p = Pi(d,v);
        
    else 
        
        % primera iteración 
        d = v;
        p = d;
    end
    
    alpha = 1;
    % línea 10 ( recorte del paso )  línea 12 código
    [xnew, alpha] = Armijo(X, ET, f, x, alpha, beta, gam, v, d, p, maxrec, lambda);
    
    % 
    % ... almacenamos (s,y) 25-26 códgio
    s  = xnew - x;
    x  = xnew;
    f0 = f;
    g0 = g;
    
    % actualizamos
    [ f, g ] = feval( 'FunDer', X, ET, x);
    % regularización con norma 1
    f = f + lambda*norm(x,1);
    fun = [fun f];
    dif_f = abs(f0 - f)/f0;
    y = g - g0;
    
    a = mod(iter,m) + 1;
    ind(a) = iter + 1;
    S(:,a) = s;
    Y(:,a) = y;
    sTy    = s'*y;
    
    if sTy <= eps
        fprintf(' Warning, small sTy < eps  % 8.2e \n', sTy);
    end
   
    gamma_k  = (sTy / (y' * y));
    norm_v = norm(v,inf); 
    iter   = iter + 1;
    
    %
    psgrad = [psgrad norm_v/(1.0+normv0)];
    nonzero = [nonzero nnz(x)];

    if mod(iter,impr) == 0
       fprintf(' %3i  %16.10e   %8.2e   %8.5e   %5.3f    %5s  \n', iter, f, norm_v/(1.0+normv0),dif_f, alpha, sTy );
    end
end
toc
end

%-------------------------------------------------------------------
%-------------------------------------------------------------------

function [xnew,step] = Armijo( A, L,f, x,alpha, beta, gam,v,d,p, maxrec,lambda)
    %[xnew, alpha] = Armijo(X, ET, f, x, alpha, beta, gam, v, d, p, maxrec, lambda);
    %contador de recortes
    rec = 0;
    
    % Ecuación (3) del artículo para restringir el ortante
    xi  =  Xi( x, v);
    
    % Para mantener el nuevo punto en el mismo ortante
    % función Pi en la sección 2.1 del artículo
   
    x_a   = Pi(x + alpha*p, xi);
    
    % f evaluada en el nuevo punto con regularización
    [fn,~] =  feval('FunDer', A, L, x_a);
    fn = fn + lambda*norm(x_a,1);
    
    % recortamos alpha mientras no se cumpla la condición
    % ecuación (7) del artículo en la sección 3
    while ((fn > f - gam*alpha*(v'*d)) &&  rec < maxrec)
        
        % actualizar como antes
        alpha = alpha*beta; 
        x_a     = Pi( x+alpha*p, xi);
        [fn,~] =  feval( 'FunDer', A, L, x_a);
        fn    = fn + lambda*norm(x_a,1);
        rec   = rec+1;
    end
    xnew = x_a;
    step=alpha;


end

%-------------------------------------------------------------------
%-------------------------------------------------------------------

function [ f, g ] = FunDer( X, Y, w)
%     F es l, g es la derivada de l
%     X n x p matrix containing the samples
%     Y is the vector of labels  {0,1}. 
%     w is the vector of parameters (las betas)
% -------------------------------------------------------------------
%
[ N, p ] = size(X);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Con etiquetas en {1,-1}
aux = -Y.*(X*w);
Log = log(1+exp(aux));
Log(aux>709) = aux(aux>709);
f = mean( Log );

% ... gradient of the loss function
term =   1./(1 + exp(aux));
g = zeros(p,1);
g_aux = (-Y.*exp(aux)).*term;
g(1) = sum(g_aux); 
for i=2:p
    g(i) = sum( ( g_aux ).*X(:,i) );
end
g = (1/N)*g;
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

function [r]= loop_LBFGS(s, y, g, gamma, indice)
    % Nocedal 196->
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
    for i = 1:m
       ro(a) = 1 / (y(:,a)' * s(:,a));
       a = mod(a,m) + 1;
    end
    a = bm;
    alfa=zeros(m,1);
    for i=1:m
        alfa(a)= ro(a) * s(:,a)' * q;
        q = q - alfa(a) * y(:,a);
        a = mod(a - 2,m) + 1;
    end
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

