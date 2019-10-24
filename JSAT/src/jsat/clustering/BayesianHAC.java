/*
 * This code was contributed under the Public Domain
 */
package jsat.clustering;

import java.util.Arrays;
import java.util.Collection;
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
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.Stack;
import java.util.function.IntSupplier;
import java.util.stream.IntStream;

/**
 *
 * @author Edward Raff
 */
public class BayesianHAC implements Clusterer
{
    private double alpha_prior = 1.0;
    private Distributions dist = Distributions.BERNOULLI_BETA;

    static public enum Distributions 
    {
        BERNOULLI_BETA
        {
            @Override
            public Node init(int point, double alpha_prior, List<Vec> data) 
            {
                return new BernoulliBetaNode(point, alpha_prior, data);
            }
        };
        
        abstract  public Node init(int point, double alpha_prior, List<Vec> data);
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
        double d;
        double pi_k;
        
        /**
         * Stores the value of p(D_k | T_k), assuming this current node is (D_k | T_k)
         */
        double log_pdt;
        
        Distribution left_child;
        Distribution right_child;
        
        int size;
        
        

        public Node(int single_point, double alpha_prior) //used for base case init
        {
            this.owned = single_point;
            this.log_pdt = 1;
            this.size = 1;
            //﻿initialize each leaf i to have d_i = α, π_i = 1
            this.d = alpha_prior;
            this.pi_k = 1.0;
        }
        
        public Node(Distribution a, Distribution b, double alpha_prior) //MERGE THE NODES
        {
            this.owned = -1;
            this.log_pdt = Double.NaN;
            this.size = a.size + b.size;
            
            //﻿Figure 3. of paper for equations
            double tmp = alpha_prior * gamma(this.size);
            this.d = tmp + a.d * b.d;
            this.pi_k = tmp/this.d;
            
            this.left_child = a;
            this.right_child = b;
        }
        
        public double r(List<Vec> dataset, HyperParams priors)
        {
            if(this.size == 1)
            {
                //get this computed for future please
                this.log_pdt = this.log_null(dataset, priors);
                return 1.0;
            }
            
            double log_pi = log(this.pi_k);
            double log_numer = log_pi+this.log_null(dataset, priors);
            //rhight hand side of equation 2
            double log_rhs = log(1-this.pi_k) + left_child.log_pdt + right_child.log_pdt;
            
            this.log_pdt = log_exp_sum(log_numer, log_rhs);
            
            return Math.exp(log_numer-this.log_pdt);
        }

        abstract public Distribution merge(Distribution a, Distribution b, double alpha_prior);
        
        abstract public HyperParams computeInitialPrior(List<Vec> dataset);
        
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

                
    }
            
    public BayesianHAC() 
    {
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
            n.r(data, priors);
            current_nodes.add(n);
        }
        
        
        while(current_nodes.size() > 1)
        {
            double best_r = 0;
            int best_i = -1, best_j = -1;
            Node best_merged = null;
            
            for(int i = 0; i < current_nodes.size(); i++)
            {
                Node D_i = current_nodes.get(i);
                        
                for(int j = i+1; j < current_nodes.size(); j++)
                {
                    Node D_j = current_nodes.get(j);
                    
                    Node merged = D_i.merge(D_i, D_j, alpha_prior);
                    double r = merged.r(data, priors);
                    
//                    System.out.println(r + "," + D_i.owned + "," + D_j.owned);
                    
                    if(r > best_r)
                    {
                        best_i = i;
                        best_j = j;
                        best_merged = merged;
                        best_r = r;
                    }
                }
            }
            
            if(best_r > 0.5)
            {
//                System.out.println(best_r + " merge " + current_nodes.get(best_i).owned + " " + current_nodes.get(best_j).owned);
                current_nodes.remove(best_j);
                current_nodes.remove(best_i);
                current_nodes.add(best_merged);
            }
            else
                break;
        }
        
        for(int class_id = 0; class_id < current_nodes.size(); class_id++)
        {
            Iterator<Integer> owned = current_nodes.get(class_id).indxIter();
            
//            System.out.println(current_nodes.get(class_id).size);
//            System.out.print(class_id + ":");
            while(owned.hasNext())
            {
                int pos = owned.next();
//                System.out.print(pos + "," );
                designations[pos] = class_id;
            }
//            System.out.println();
        }
        
        return designations;
    }

    @Override
    public Clusterer clone() {
        return this;
    }
    
}
