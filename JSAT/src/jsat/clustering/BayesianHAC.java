/*
 * This code was contributed under the Public Domain
 */
package jsat.clustering;

import java.util.List;
import jsat.DataSet;
import jsat.linear.ConstantVector;
import jsat.linear.DenseVector;
import jsat.linear.MatrixStatistics;
import jsat.linear.Vec;
import jsat.utils.IntList;
import static  jsat.math.SpecialMath.*;
import static  java.lang.Math.log;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.Stack;
import java.util.stream.Collectors;
import jsat.distributions.Distribution;
import jsat.distributions.discrete.Binomial;
import jsat.distributions.multivariate.IndependentDistribution;
import jsat.distributions.multivariate.MultivariateDistribution;
import jsat.distributions.multivariate.NormalM;
import jsat.linear.CholeskyDecomposition;
import jsat.linear.DenseMatrix;
import jsat.linear.Matrix;
import jsat.linear.SingularValueDecomposition;
import jsat.math.OnLineStatistics;

/**
 *
 * @author Edward Raff
 */
public class BayesianHAC implements Clusterer
{
    private double alpha_prior = 1.0;
    private Distributions dist = Distributions.BERNOULLI_BETA;
    
    /**
     * After clustering, one possibility is to interpret each found cluster as
     * its own separate distribution. This list stores the results of that
     * interpretation.
     */
    protected List<MultivariateDistribution> cluster_dists;

    static public enum Distributions 
    {
        BERNOULLI_BETA
        {
            @Override
            public Node init(int point, double alpha_prior, List<Vec> data) 
            {
                return new BernoulliBetaNode(point, alpha_prior, data);
            }
        },
        GAUSSIAN_DIAG
        {
            @Override
            public Node init(int point, double alpha_prior, List<Vec> data) 
            {
                return new NormalDiagNode(point, alpha_prior, data);
            }
        },
        GAUSSIAN_FULL
        {
            @Override
            public Node init(int point, double alpha_prior, List<Vec> data) 
            {
                return new NormalNode(point, alpha_prior, data);
            }
        };
        
        abstract  public Node init(int point, double alpha_prior, List<Vec> data);
    }
    
    public BayesianHAC() 
    {
        this(Distributions.GAUSSIAN_DIAG);
    }
    
    public BayesianHAC(Distributions dist) 
    {
        this.dist = dist;
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public BayesianHAC(BayesianHAC toCopy) 
    {
        this.alpha_prior = toCopy.alpha_prior;
        this.dist = toCopy.dist;
        if(toCopy.cluster_dists != null)
            this.cluster_dists = toCopy.cluster_dists.stream()
                    .map(MultivariateDistribution::clone).collect(Collectors.toList());
    }
    
    
    
    
    /**
     * Computes log(exp(a)+exp(b)) in an accurate manner
     * @param log_a
     * @param log_b
     * @return 
     */
    static double log_exp_sum(double log_a, double log_b)
    {
        if(log_b > log_a)
            return log_exp_sum(log_b, log_a);
        //if a > b, use identify 
        //log(a+b) = log(a) + log(1 + a/b)
        //left hand side, since we do exp first, we get
        //log(exp(a)) = a, so nothing changes
        //right hand side, log(1+ exp(b)/exp(a)) = log(exp(b-a)+ 1)
        
        return log_a + Math.log1p(Math.exp(log_b-log_a));
    }
    
    protected static interface DistPrior
    {
        
    };
    
    protected static abstract class Node<Distribution extends Node, HyperParams extends DistPrior>
    {
        int owned;
        IntList allChilds;
        double log_d;
        double log_pi;
        
        /**
         * Stores the value of p(D_k | T_k), assuming this current node is (D_k | T_k)
         */
        double log_pdt;
        
        Distribution left_child;
        Distribution right_child;
        
        /**
         * How many data points belong to this node (inclusive) . 
         */
        int size;
        
        

        public Node(int single_point, double alpha_prior) //used for base case init
        {
            this.owned = single_point;
            this.allChilds = IntList.view(new int[]{single_point});
            this.log_pdt = 1;
            this.size = 1;
            //﻿initialize each leaf i to have d_i = α, π_i = 1
            this.log_d = log(alpha_prior);
            this.log_pi = log(1.0);
        }
        
        public Node(Distribution a, Distribution b, double alpha_prior) //MERGE THE NODES
        {
            this.owned = -1;
            this.log_pdt = Double.NaN;
            this.size = a.size + b.size;
            this.allChilds = new IntList(a.allChilds);
            this.allChilds.addAll(b.allChilds);
            Collections.sort(allChilds);
            
            //﻿Figure 3. of paper for equations
//            double tmp = alpha_prior * gamma(this.size);
//            this.d = tmp + a.log_d * b.log_d;
//            this.pi = tmp/this.log_d;
            double tmp = log(alpha_prior) + lnGamma(this.size);
            this.log_d = log_exp_sum(tmp, a.log_d+b.log_d);
            this.log_pi = tmp - this.log_d;
            
            this.left_child = a;
            this.right_child = b;
        }
        
        public double logR(List<Vec> dataset, HyperParams priors)
        {
            if(this.size == 1)
            {
                //get this computed for future please
                this.log_pdt = this.log_null(dataset, priors);
                return 1.0;
            }
            
//            double log_pi = log(this.log_pi);
            double log_numer = log_pi+this.log_null(dataset, priors);
            //rhight hand side of equation 2
            double log_neg_pi = log(-Math.expm1(log_pi));
            double log_rhs = log_neg_pi+ left_child.log_pdt + right_child.log_pdt;
            
            this.log_pdt = log_exp_sum(log_numer, log_rhs);
            
//            return Math.exp(log_numer-this.log_pdt);
            return log_numer-this.log_pdt;
        }

        abstract public Distribution merge(Distribution a, Distribution b, double alpha_prior);
        
        abstract public HyperParams computeInitialPrior(List<Vec> dataset);
        
        /**
         * Interpreting the current node as a cluster, this method should return
         * a multivariate distribution object that summarizes the content of
         * this node, ignoring the rest of the tree.
         *
         * @param dataset the original training dataset in the original order
         * @return a distribution object representing this node.
         */
        abstract public MultivariateDistribution toDistribution(List<Vec> dataset);
        
        public boolean isLeaf()
        {
            return right_child == null && left_child == null;
        }
        
        /**
         * Computes the log(﻿p(D|H_1)) for the current distribution assumption
         * @param dataset
         * @param priors
         * @return 
         */
        abstract public double log_null(List<Vec> dataset, HyperParams priors);
        
        public Iterator<Integer> indxIter()
        {
            Stack<Node> remains = new Stack<>();
            remains.push(this);
            
            return new Iterator<Integer>() 
            {
                @Override
                public boolean hasNext() 
                {
                    while(!remains.isEmpty() && !remains.peek().isLeaf())
                    {
                        Node c = remains.pop();
                        remains.push(c.left_child);
                        remains.push(c.right_child);
                    }
                    
                    return !remains.empty();
                }

                @Override
                public Integer next() 
                {
                    Node c = remains.pop();
                    return c.owned;
                }
            };
        }
        
        public List<Integer> ownedList()
        {
            IntList a = new IntList(this.size);
            Iterator<Integer> iter = this.indxIter();
            while(iter.hasNext())
                a.add(iter.next());
            return a;
        }
    }
    
    protected static class BetaConjugate implements DistPrior
    {
        public Vec alpha_prior;
        public Vec beta_prior;

        public BetaConjugate(List<Vec> dataset) 
        {
            int d = dataset.get(0).length();
            
            Vec mean = MatrixStatistics.meanVector(dataset);
            
            alpha_prior = mean.multiply(2).add(1e-3);
            
            beta_prior = new DenseVector(new ConstantVector(1, d)).subtract(mean).multiply(2).add(1e-3);
        }
    }
    
    protected static class WishartDiag implements DistPrior
    {
        /**
         * ﻿v is the degree of freedom
         */
        double v;
        /**
         * ﻿r is scaling factor on the prior precision of the mean,
         */
        double r;
        
        /**
         * ﻿m which is the prior on the mean
         */
        Vec m;
        /**
         * ﻿S is the prior on the precision matrix.
         * In our case, S is the diag of it. 
         * 
         */
        Vec S;
        
        double log_shared_term;
        
        public WishartDiag(List<Vec> dataset) 
        {
            int N = dataset.size();
            int k = dataset.get(0).length();
            v = k;
            
            r = 0.001;
            
            m = new DenseVector(k);
            MatrixStatistics.meanVector(m, dataset);
            
            S = new DenseVector(k);
            MatrixStatistics.covarianceDiag(m, S, dataset);
            S.mutableDivide(20);

            
            //Lets get the last term with the prod in it first b/c it contains 
            //many additions and subtractions
            log_shared_term = 0;
            

            double log_det_S = 0;
            
            for(int i = 0; i < k; i++)
                log_det_S += log(S.get(i));
            log_shared_term += v/2*log_det_S;
            
        }
    }
    
    protected static class WishartFull implements DistPrior
    {
        /**
         * ﻿v is the degree of freedom
         */
        double v;
        /**
         * ﻿r is scaling factor on the prior precision of the mean,
         */
        double r;
        
        /**
         * ﻿m which is the prior on the mean
         */
        Vec m;
        /**
         * ﻿S is the prior on the precision matrix.
         * In our case, S is the diag of it. 
         * 
         */
        Matrix S;
        
        double log_shared_term;
        
        public WishartFull(List<Vec> dataset) 
        {
            int N = dataset.size();
            int k = dataset.get(0).length();
            v = k;
            
            r = 0.001;
            
            m = new DenseVector(k);
            MatrixStatistics.meanVector(m, dataset);
            
            S = new DenseMatrix(k, k);
            MatrixStatistics.covarianceMatrix(m, S, dataset);
            

            SingularValueDecomposition svd = new SingularValueDecomposition(S.clone());
            if(svd.isFullRank())
            {
                S.mutableMultiply(1.0/20);
            }
            else
            {
                OnLineStatistics var = new OnLineStatistics();
                for(Vec v : dataset)
                    for(int i = 0; i < v.length(); i++)
                        var.add(v.get(i));

                for(int i = 0; i < S.rows(); i++)
                    S.increment(i, i, 0.1*S.get(i, i) + var.getVarance());
            }

            
            //Lets get the last term with the prod in it first b/c it contains 
            //many additions and subtractions
            log_shared_term = 0;
            
            CholeskyDecomposition cd = new CholeskyDecomposition(S.clone());
            double log_det_S = cd.getLogDet();
            log_shared_term += v/2*log_det_S;
            
        }
    }
    
    protected static class BernoulliBetaNode extends Node<BernoulliBetaNode, BetaConjugate>
    {
        public Vec m;
        
        public BernoulliBetaNode(int single_point, double alpha_prior, List<Vec> dataset) 
        {
            super(single_point, alpha_prior);
            this.m = dataset.get(single_point);
        }

        public BernoulliBetaNode(BernoulliBetaNode a, BernoulliBetaNode b, double alpha_prior) 
        {
            super(a, b, alpha_prior);
            this.m = a.m.add(b.m);
        }

        @Override
        public BetaConjugate computeInitialPrior(List<Vec> dataset)
        {
            return new BetaConjugate(dataset);
        }

        @Override
        public double log_null(List<Vec> dataset, BetaConjugate priors) 
        {
            Vec alpha = priors.alpha_prior;
            Vec beta = priors.beta_prior;
            int N = this.size;
            int D = dataset.get(0).length();
            double log_prob = 0;
            for(int d = 0; d < D; d++)
            {
                double a_d = alpha.get(d);
                double b_d = beta.get(d);
                double m_d = this.m.get(d);
                double log_numer = lnGamma(a_d + b_d) + lnGamma(a_d + m_d) + lnGamma(b_d + N - m_d);
                double log_denom = lnGamma(a_d) + lnGamma(b_d) + lnGamma(a_d + b_d + N);
                
                log_prob += (log_numer - log_denom);
            }
            return log_prob;
        }

        @Override
        public BernoulliBetaNode merge(BernoulliBetaNode a, BernoulliBetaNode b, double alpha_prior) 
        {
            return new BernoulliBetaNode(a, b, alpha_prior);
        }

        @Override
        public MultivariateDistribution toDistribution(List<Vec> dataset) 
        {
            //TODO add Bernoulli option and use that. But Binomial with 1 trial is equivalent
            List<Distribution> dists = new ArrayList<>();
            double N = this.size;
            for(int i = 0; i < m.length(); i++)
                dists.add(new Binomial(1, m.get(i)/N));
            
            return new IndependentDistribution(dists);
        }
    }
    
    protected static class NormalDiagNode extends Node<NormalDiagNode, WishartDiag>
    {
        /**
         * X X^T term is really
         * X^T X for row, col format like us. 
         * In diag case, diag(X^T X)_j = \sum_i X_ij^2
         */
        Vec XT_X;
        
        Vec x_sum;
        

        public NormalDiagNode(int single_point, double alpha_prior, List<Vec> dataset) 
        {
            super(single_point, alpha_prior);
            
            Vec x_i = dataset.get(single_point);
            this.XT_X = x_i.pairwiseMultiply(x_i);
            this.x_sum = x_i;
        }
        
        public NormalDiagNode(NormalDiagNode a, NormalDiagNode b, double alpha_prior) 
        {
            super(a, b, alpha_prior);
            this.XT_X= a.XT_X.add(b.XT_X);
            this.x_sum = a.x_sum.add(b.x_sum);
        }

        @Override
        public NormalDiagNode merge(NormalDiagNode a, NormalDiagNode b, double alpha_prior) 
        {
            NormalDiagNode node = new NormalDiagNode(a, b, alpha_prior);
            
            return node;
        }

        @Override
        public WishartDiag computeInitialPrior(List<Vec> dataset) 
        {
            return new WishartDiag(dataset);
        }

        @Override
        public MultivariateDistribution toDistribution(List<Vec> dataset) 
        {
            List<Integer> ids = this.ownedList();
            Vec mean = new DenseVector(dataset.get(0).length());
            MatrixStatistics.meanVector(mean, dataset, ids);
            Vec cov = new DenseVector(mean.length());
            MatrixStatistics.covarianceDiag(mean, cov, dataset, ids);
            
            return new NormalM(mean, cov);
        }

        @Override
        public double log_null(List<Vec> dataset, WishartDiag priors) 
        {
            int N = this.size;
            double r = priors.r;
            int k = priors.m.length();
            double v = priors.v;
            
            
            //init with first two terms
            Vec S_prime = priors.S.add(this.XT_X);
            //
            // m m^T is the outer-product, but we are just diag, 
            //so diag(m m^T)_j = m_j^2
            
            Vec mm = priors.m.pairwiseMultiply(priors.m);
            
            S_prime.mutableAdd(r*N/(N+r), mm);
            
            // diag((\sum x) (\sum x)^T )_i = (\sum x)_i^2
            
            Vec xsum_xsum = x_sum.pairwiseMultiply(x_sum);
            
            S_prime.mutableAdd(-1/(N+r), xsum_xsum);
            
            //diag((m * xsum^T + xsum * m^T))_i = m_i * xsum_i * 2
            
            Vec mxsum = priors.m.pairwiseMultiply(x_sum).multiply(2);
            
            S_prime.mutableAdd(-r/(N+r), mxsum);
            
            
            double v_p = priors.v + N;
            
            double log_det_S_p = 0;
            for(int i = 0; i < S_prime.length(); i++)
                log_det_S_p += log(S_prime.get(i));
            
            double log_prob = priors.log_shared_term + -v_p/2*log_det_S_p;
            
            for(int j = 1; j <= k; j++)
                log_prob += lnGamma((v_p+1-j)/2) - lnGamma((v+1-j)/2);
            log_prob += v_p*k/2.0*log(2) - v*k/2.0*log(2);
            
            log_prob += -N*k/2.0*log(2*Math.PI);

            log_prob += k/2.0 * (log(r) - log(N+r));

            return log_prob;
        }
        
    }
    
    protected static class NormalNode extends Node<NormalNode, WishartFull>
    {
        /**
         * X^T X for row, col format like us. 
         * For incremental updates of X^T X, when we add a new row z, it becomes
         * X^T X  + z^T z, so we just add the outer product update to X^T X
         * 
         */
        Matrix XT_X;
        
        Vec x_sum;
        

        public NormalNode(int single_point, double alpha_prior, List<Vec> dataset) 
        {
            super(single_point, alpha_prior);
            
            Vec x_i = dataset.get(single_point);
            this.XT_X = new DenseMatrix(x_i.length(), x_i.length());
            Matrix.OuterProductUpdate(XT_X, x_i, x_i, 1.0);
            this.x_sum = x_i;
        }
        
        public NormalNode(NormalNode a, NormalNode b, double alpha_prior) 
        {
            super(a, b, alpha_prior);
            this.XT_X= a.XT_X.add(b.XT_X);
            this.x_sum = a.x_sum.add(b.x_sum);
            
        }

        @Override
        public NormalNode merge(NormalNode a, NormalNode b, double alpha_prior) 
        {
            NormalNode node = new NormalNode(a, b, alpha_prior);
            
            return node;
        }

        @Override
        public WishartFull computeInitialPrior(List<Vec> dataset) 
        {
            return new WishartFull(dataset);
        }
        
        @Override
        public MultivariateDistribution toDistribution(List<Vec> dataset) 
        {
            List<Integer> ids = this.ownedList();
            Vec mean = new DenseVector(dataset.get(0).length());
            MatrixStatistics.meanVector(mean, dataset, ids);
            Matrix cov = new DenseMatrix(mean.length(), mean.length());
            MatrixStatistics.covarianceMatrix(mean, cov, dataset, ids);
            
            return new NormalM(mean, cov);
        }

        @Override
        public double log_null(List<Vec> dataset, WishartFull priors) 
        {
            int N = this.size;
            double r = priors.r;
            int k = priors.m.length();
            double v = priors.v;
            
            
            //init with first two terms
            Matrix S_prime = priors.S.add(this.XT_X);
            //
            // m m^T is the outer-product,
            
            Matrix.OuterProductUpdate(S_prime, priors.m, priors.m, r*N/(N+r));
            
            //4th term, outer product update of row sums
            Matrix.OuterProductUpdate(S_prime, x_sum, x_sum, -1/(N+r));
            
            //-r/(N+r) (m * xsum^T + xsum * m^T), lets break it out into two outer
            //product updates, 
            
            Matrix.OuterProductUpdate(S_prime, priors.m, x_sum, -r/(N+r));
            Matrix.OuterProductUpdate(S_prime, x_sum, priors.m, -r/(N+r));
            
            
            double v_p = priors.v + N;
            
            CholeskyDecomposition cd = new CholeskyDecomposition(S_prime);
            double log_det_S_p = cd.getLogDet();
            
            double log_prob = priors.log_shared_term + -v_p/2*log_det_S_p;
            
            for(int j = 1; j <= k; j++)
                log_prob += lnGamma((v_p+1-j)/2) - lnGamma((v+1-j)/2);
            log_prob += v_p*k/2.0*log(2) - v*k/2.0*log(2);
            
            log_prob += -N*k/2.0*log(2*Math.PI);

            log_prob += k/2.0 * (log(r) - log(N+r));

            return log_prob;
        }
        
    }
            

    @Override
    public int[] cluster(DataSet dataSet, boolean parallel, int[] designations) 
    {
        List<Vec> data = dataSet.getDataVectors();
        
        if(designations == null)
            designations = new int[data.size()];
        
        
        DistPrior priors = null;
        List<Node> current_nodes = new ArrayList<>();
        for(int i = 0; i < data.size(); i++)
        {
            Node n = dist.init(i, alpha_prior, data);
            if(priors == null)
                priors = n.computeInitialPrior(data);
            n.logR(data, priors);
            current_nodes.add(n);
        }
        
        
        while(current_nodes.size() > 1)
        {
            double best_r = Double.NEGATIVE_INFINITY;
            int best_i = -1, best_j = -1;
            Node best_merged = null;
            
            for(int i = 0; i < current_nodes.size(); i++)
            {
                Node D_i = current_nodes.get(i);
                        
                for(int j = i+1; j < current_nodes.size(); j++)
                {
                    Node D_j = current_nodes.get(j);
                    
                    Node merged = D_i.merge(D_i, D_j, alpha_prior);
                    double log_r = merged.logR(data, priors);
                    
//                    System.out.println("\t" + log_r + "," + D_i.allChilds + "," + D_j.allChilds);
                    
                    if(log_r > best_r)
                    {
                        best_i = i;
                        best_j = j;
                        best_merged = merged;
                        best_r = log_r;
                    }
                }
            }
            
//            System.out.println(Math.exp(best_r) + " merge " + current_nodes.get(best_i).allChilds + " " + current_nodes.get(best_j).allChilds + " | " + best_merged.log_pi);
            if(best_r > log(0.5))
            {
                current_nodes.remove(best_j);
                current_nodes.remove(best_i);
                current_nodes.add(best_merged);
            }
            else
                break;
        }
        
//        System.out.println("C: " + current_nodes.size());
        
        this.cluster_dists = new ArrayList<>(current_nodes.size());
        for(int class_id = 0; class_id < current_nodes.size(); class_id++)
        {
            List<Integer> owned = current_nodes.get(class_id).ownedList();
                    
            
//            System.out.println(current_nodes.get(class_id).size);
//            System.out.print(class_id + ":");
            for(int pos : owned)
                designations[pos] = class_id;
//            System.out.println();
            this.cluster_dists.add(current_nodes.get(class_id).toDistribution(data));
        }
        
        return designations;
    }
    
    public List<MultivariateDistribution> getClusterDistributions()
    {
        return cluster_dists;
    }

    @Override
    public BayesianHAC clone() 
    {
        return this;
    }
    
}
