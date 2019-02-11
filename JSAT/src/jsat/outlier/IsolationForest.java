/*
 * Copyright (C) 2018 edraff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jsat.outlier;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.math.FastMath;
import jsat.math.SpecialMath;
import jsat.utils.IntList;
import jsat.utils.concurrent.ParallelUtils;
import jsat.utils.random.RandomUtil;

/**
 *
 * @author edraff
 */
public class IsolationForest implements Outlier
{
    private int trees = 100;
    private double subSamplingSize = 256;
    
    List<iTreeNode> roots = new ArrayList<>();

    public IsolationForest()
    {
    }
    
    public IsolationForest(IsolationForest toCopy)
    {
        this.trees = toCopy.trees;
        this.subSamplingSize = toCopy.subSamplingSize;
        this.roots = new ArrayList<>();
        for(iTreeNode root : toCopy.roots)
            this.roots.add(root.clone());
    }
    
    /**
     * Equation 1 in the Isolation Forest paper
     * @param n
     * @return 
     */
    private static double c(double n)
    {
        return 2*SpecialMath.harmonic(n-1)-(2*(n-1)/n);
    }

    @Override
    public void fit(DataSet d, boolean parallel)
    {
        for(int i =0; i < trees; i++)
            roots.add(new iTreeNode());
        
        int l = (int) Math.ceil(Math.log(subSamplingSize)/Math.log(2));
        int D = d.getNumNumericalVars();
        //Build all the trees
        ParallelUtils.streamP(roots.stream(), parallel).forEach(r->
        {
            r.build(0, l, d, IntList.range(d.size()), new double[D], new double[D]);
        });
    }

    @Override
    public double score(DataPoint x)
    {
        double e_h_x = roots.stream().
                mapToDouble(r->r.pathLength(x.getNumericalValues(), 0))
                .average().getAsDouble();
        //an anomaly score is produced by computing s(x,ψ) in Equation 2
        
        double anomScore = FastMath.pow2(-e_h_x/c(subSamplingSize));
        //anomScore will be in the range [0, 1]
        //values > 0.5 are considered anomolies
        //so just return 0.5-anomScore to fit the interface defintion
        return 0.5-anomScore;
    }

    @Override
    protected IsolationForest clone() throws CloneNotSupportedException
    {
        return new IsolationForest(this);
    }
    
    private class iTreeNode implements Serializable
    {
        
        iTreeNode leftChild;
        iTreeNode rightChild;
        double size = 0;
        double splitVal;
        int splitAtt;
        

        public iTreeNode()
        {
        }

        public iTreeNode(iTreeNode toCopy)
        {
            this.leftChild = new iTreeNode(toCopy.leftChild);
            this.rightChild = new iTreeNode(toCopy.rightChild);
            this.splitVal = toCopy.splitVal;
            this.splitAtt = toCopy.splitAtt;
        }

        @Override
        protected iTreeNode clone() 
        {
            return new iTreeNode(this);
        }
        
        public void build(int e, int l, DataSet source, IntList X, double[] minVals, double[] maxVals)
        {
            if(e >= l || X.size() <= 1)
            {
                if(X.isEmpty())//super rare, rng guesses the min value itself
                    this.size = 1;
                else//use average
                    this.size = X.stream().mapToDouble(s->source.getWeight(s)).sum();
                    
                //else, size stays zero
                return;
            }
            //else
            
            
            int D = source.getNumNumericalVars();
            Arrays.fill(minVals, 0.0);
            Arrays.fill(maxVals, 0.0);
                        
            //find the min-max range for each feature
            X.stream().forEach(d->
            {
                for(IndexValue iv : source.getDataPoint(d).getNumericalValues())
                {
                    int i = iv.getIndex();
                    minVals[i] = Math.min(minVals[i], iv.getValue());
                    maxVals[i] = Math.max(maxVals[i], iv.getValue());
                }
            });
            
            //how many features are valid choices?
            int candiadates = 0;
            for(int i = 0; i < D; i++)
                if(minVals[i] != maxVals[i])
                    candiadates++;
            //select the q'th feature with a non-zero spread
            int q_candidate = RandomUtil.getLocalRandom().nextInt(candiadates);
            int q = 0;
            for(int i = 0; i <D; i++)
                if(minVals[i] != maxVals[i])
                    if(--q_candidate == 0)
                    {
                        q = i;
                        break;
                    }
            
            //pick random split value between min & max
            splitVal = RandomUtil.getLocalRandom().nextDouble();
            splitVal = minVals[q] + (maxVals[q]-minVals[q])*splitVal;
            
            IntList X_l = new IntList();
            IntList X_r = new IntList();
            for(int x : X)
                if(source.getDataPoint(x).getNumericalValues().get(q) < splitVal)
                    X_l.add(x);
                else
                    X_r.add(x);
            splitAtt = q;
            
            this.leftChild = new iTreeNode();
            this.leftChild.build(e+1, l, source, X_l, minVals, maxVals);
            this.rightChild = new iTreeNode();
            this.rightChild.build(e+1, l, source, X_r, minVals, maxVals);
        }
        
        
        public double pathLength(Vec x, double e)
        {
            if(leftChild == null)//root node
                return e + c(this.size);
            //else
            if(x.get(splitAtt) < splitVal)
                return leftChild.pathLength(x, e+1);
            else
                return rightChild.pathLength(x, e+1);
        }
    }
}
