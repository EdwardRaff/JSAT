/*
 * Copyright (C) 2015 Edward Raff
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
package jsat.datatransform.visualization;

import java.util.*;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.datatransform.*;
import jsat.distributions.Normal;
import jsat.linear.*;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.vectorcollection.VPTreeMV;
import jsat.math.FastMath;
import jsat.math.FunctionBase;
import jsat.math.optimization.stochastic.*;
import jsat.math.rootfinding.Zeroin;
import jsat.utils.random.XORWOW;

/**
 *
 * @author Edward Raff
 */
public class TSNE
{
    public double alpha = 4;
    public double exageratedPortion = 0.25;
    DistanceMetric dm = new EuclideanDistance();
    public int T = 1000;
    public double perplexity = 30;
    double theta = 0.5;
    /**
     * The target embedding dimension, hard coded to 2 for now
     */
    int s = 2;
    
    public <Type extends DataSet> Type transform(DataSet<Type> d)
    {
        Random rand = new XORWOW(123);
        final int N = d.getSampleSize();
        //If perp set too big, the search size would be larger than the dataset size. So min to N
        /**
         * form sec 4.1: "we compute the sparse approximation by finding the
         * floor(3u) nearest neighbors of each of the N input objects (recall
         * that u is the perplexity of the conditional distributions)"
         */
        final int knn = (int) Math.min(Math.floor(3*perplexity), N);
        
        
        
        final List<Vec> vecs = d.getDataVectors();
        final List<Double> accelCache = dm.getAccelerationCache(vecs);
        
        VPTreeMV<Vec> vp = new VPTreeMV<Vec>(vecs, dm);
        
        final List<List<? extends VecPaired<Vec, Double>>> neighbors = new ArrayList<List<? extends VecPaired<Vec, Double>>>(N);
        
        
        /**
         * Each row is the set of 3*u indices returned by the NN search
         */
        final int[][] nearMe = new int[N][knn];
        
        //new scope b/c I don't want to leark the silly vecIndex thing
        {
            //Used to map vecs back to their index so we can store only the ones we need in nearMe
            IdentityHashMap<Vec, Integer> vecIndex = new IdentityHashMap<Vec, Integer>(N);
            for(int i = 0; i < N; i++) {
              vecIndex.put(vecs.get(i), i);
            }
        
            for(int i = 0; i < N; i++)//lets pre-compute the 3u nearesst neighbors used in eq(1)
            {
                Vec x_i = vecs.get(i);
                List<? extends VecPaired<Vec, Double>> closest = vp.search(x_i, knn+1);//+1 b/c self is closest
                neighbors.add(closest);
                if(i % 100 == 0) {
                  System.out.println((i+1)+"/"+N);
                }
                for (int j = 1; j < closest.size(); j++) {
                  nearMe[i][j - 1] = vecIndex.get(closest.get(j).getVector());
                }
            }
        
        }
        //Now lets figure out everyone's sigmas
        double[] sigma = new double[N];
        
        double minSigma = Double.POSITIVE_INFINITY;
        double maxSigma = 0;

        for(int i = 0; i < N; i++)//first lets figure out a min/max range
        {
            double min = neighbors.get(i).get(1).getPair();
            double max = neighbors.get(i).get(knn).getPair();
            minSigma = Math.min(minSigma, min);
            maxSigma = Math.max(maxSigma, max);
        }
        
        System.out.println("Bandwidth");
        
        
        //now compute the bandwidth for each datum 
        for(int i = 0; i < N; i++)
        {
            if(i % 100 == 0) {
              System.out.println((i+1)+"/"+N);
            }
            final int I = i;

            boolean tryAgain = false;
            do
            {
                tryAgain = false;
                try
                {
                    double sigma_i = Zeroin.root(1e-1, 100, minSigma, maxSigma, 0, new FunctionBase()
                    {
                        @Override
                        public double f(Vec x)
                        {
                            return perp(I, nearMe, x.get(0), neighbors, vecs, accelCache)-perplexity;
                        }
                    });

                    sigma[i]= sigma_i;
                }
                catch(ArithmeticException ex)//perp not in search range? 
                {
                    tryAgain = true;
                    minSigma = Math.max(minSigma/2, 1e-6);
                    maxSigma = Math.min(maxSigma*2, Double.MAX_VALUE/2);
                }
            }
            while(tryAgain);
        }
        
        
        /**
         * P_ij does not change at this point, so lets compute these values only
         * once please! j index matches up to the value stored in nearMe
         */
        double[][] nearMePij = new double[N][knn];
        
        for(int i = 0; i < N; i++)
        {
            for(int j_indx = 0; j_indx < knn; j_indx++)
            {
                int j = nearMe[i][j_indx];
                nearMePij[i][j_indx] =  p_ij(i, j, sigma[i], sigma[j], neighbors, vecs, accelCache);
            }
        }
        
        Normal normalDIst = new Normal(0, 1e-4);
        /**
         * For now store all data in a 2d array to avoid excessive overhead / cache missing
         */
        final double[] y = normalDIst.sample(N*s, rand);
        
        double[] y_grad = new double[y.length];
        
        //vec wraped version for convinence 
        final Vec y_vec = DenseVector.toDenseVec(y);
        final Vec y_grad_vec = DenseVector.toDenseVec(y_grad);
        
        GradientUpdater gradUpdater = new Adam();
        gradUpdater.setup(y.length);
        
        System.out.println("GD");

        for (int iter = 0; iter < T; iter++)//optimization
        {
            if (iter % 100 == 0) {
              System.out.println((iter) + "/" + T);
            }

            Arrays.fill(y_grad, 0);
            
            //First loop for the F_rep forces, we do this first to normalize so we can use 1 work space for the gradient
            Quadtree qt = new Quadtree(y);
            double[] workSpace = new double[s];
            double Z = 0;
            for (int i = 0; i < N; i++)
            {
                Arrays.fill(workSpace, 0.0);
                Z += computeF_rep(qt.root, i, y, workSpace);

//                double cnst = 1*4;
                //should be multiplied by 4, rolling it into the normalization by Z after
                for (int k = 0; k < s; k++) {
                  inc_z_ij(workSpace[k], i, k, y_grad, s);
                }
            }
            //normalize by Z
            final double zNorm = 4.0/(Z+1e-13);
            for(int i = 0; i < y.length; i++) {
              y_grad[i] *= zNorm;
            }
            
            //This second loops computes the F_attr forces
            for (int i = 0; i < N; i++)//N
            {
                for(int j_indx = 0; j_indx < knn; j_indx ++) //O(u)
                {
                    int j = nearMe[i][j_indx];
                    if(i == j) {//this should never happen b/c we skipped that when creating nearMe
                      continue;
                    }
                    double pij = nearMePij[i][j_indx];
                    if(iter < T*exageratedPortion) {
                      pij *= alpha;
                    }
                    double cnst = pij*q_ijZ(i, j, y, s)*4;
                    
                    for(int k = 0; k < s; k++)
                    {
                        double diff = z_ij(i, k, y, s)-z_ij(j, k, y, s);
                        inc_z_ij(cnst*diff, i, k, y_grad, s);
                    }
                }
            }
            
            //now we have accumulated all gradients
            double eta = 200;
            
            gradUpdater.update(y_vec, y_grad_vec, eta);
        }
        
        
        DataSet<Type> transformed = d.shallowClone();
        
        final IdentityHashMap<DataPoint, Integer> indexMap = new IdentityHashMap<DataPoint, Integer>(N);
        for(int i = 0; i < N; i++) {
          indexMap.put(d.getDataPoint(i), i);
        }
        
        transformed.applyTransform(new DataTransform()
        {

            @Override
            public DataPoint transform(DataPoint dp)
            {
                int i = indexMap.get(dp);
                DenseVector dv = new DenseVector(s);
                for(int k = 0; k < s; k++) {
                  dv.set(k, y[i*2+k]);
                }
                
                return new DataPoint(dv, dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
            }

            @Override
            public DataTransform clone()
            {
                return this;
            }
        });
        
        return (Type) transformed;
    }
    
    /**
     * 
     * @param node the node to begin computing from
     * @param i
     * @param z
     * @param workSpace the indicies are the accumulated contribution to the
     * gradient sans multiplicative terms in the first 2 indices.
     * @return the contribution to the normalizing constant Z
     */
    private double computeF_rep(Quadtree.Node node, int i, double[] z, double[] workSpace)
    {
        if(node == null || node.N_cell == 0 || node.indx == i) {
          return 0;
        }
        /*
         * Original paper says to use the diagonal divided by the squared 2 
         * norm. This dosn't seem to work at all. Tried some different ideas 
         * with 0.5 as the threshold until I found one that worked. 
         * Squaring the values would normally not be helpful, but since we are working with tiny values it makes them smaller, making it easier to hit the go
         */
        double x = z[i*2];
        double y = z[i*2+1];
//        double r_cell = node.diagLen();
        double r_cell  = Math.max(node.maxX-node.minX, node.maxY-node.minY);
        r_cell*=r_cell;
        double mass_x = node.x_mass/node.N_cell;
        double mass_y = node.y_mass/node.N_cell;
        double dot = (mass_x-x)*(mass_x-x)+(mass_y-y)*(mass_y-y);
        

        if(node.NW == null || r_cell < theta*dot)//good enough! 
        {
            if(node.indx == i) {
              return 0;
            }
            
            double Z = 1.0/(1.0 + dot);
            double q_cell_Z_sqrd = -node.N_cell*(Z*Z);
            
            workSpace[0] += q_cell_Z_sqrd*(x-mass_x);
            workSpace[1] += q_cell_Z_sqrd*(y-mass_y);
            return Z*node.N_cell;
        }
        else//further subdivide
        {
            double Z_sum = 0;
            for(Quadtree.Node child : node) {
              Z_sum += computeF_rep(child, i, z, workSpace);
            }
            return Z_sum;
        }
    }
    
    /**
     * 
     * @param val the value to add to the array
     * @param i the index of the data point to add to 
     * @param j the dimension index of the embedding  
     * @param z the storage of the embedded vectors
     * @param s the dimension of the embedding
     */
    private static void inc_z_ij(double val, int i, int j, double[] z, int s)
    {
        z[i*s+j] += val;
    }
    
    private static double z_ij(int i, int j, double[] z, int s)
    {
        return z[i*s+j];
    }
    
    /**
     * Computes the value of q<sub>ij</sub> Z 
     * @param i
     * @param j
     * @param z
     * @param s
     * @return 
     */
    private static double q_ijZ(int i, int j, double[] z, int s)
    {
        double denom =1;
        for(int k = 0; k < s; k++)
        {
            double diff = z_ij(i, k, z, s)-z_ij(j, k, z, s);
            denom += diff*diff;
        }
        
        return 1.0/denom;
    }
    
    /**
     * Computes p<sub>j|i</sub>
     * @param j
     * @param i
     * @param sigma
     * @param neighbors
     * @return 
     */
    private double p_j_i(int j, int i, double sigma, List<List<? extends VecPaired<Vec, Double>>> neighbors, List<Vec> vecs, List<Double> accelCache)
    {
        /*
         * "Because we are only interested in modeling pairwise similarities, we
         * set the value of pi|i to zero" from Visualizing Data using t-SNE
         */
        if(i == j) {
          return 0;
        }
        //nearest is self, use taht to get indexed values
        Vec x_j = neighbors.get(j).get(0).getVector();
//        Vec x_i = neighbors.get(i).get(0).getVector();
        
        final double sigmaSqrdInv = 1/(2*(sigma*sigma));
        
        double numer = 0;
        double denom = 0;
        boolean jIsNearBy = false;
        final List<? extends VecPaired<Vec, Double>> neighbors_i = neighbors.get(i);
        for (int k = 1; k < neighbors_i.size(); k++)//SUM over k != i
        {
            VecPaired<Vec, Double> neighbor_ik = neighbors_i.get(k);
            final double d_ik = neighbor_ik.getPair();
            denom += FastMath.exp(-(d_ik*d_ik)*sigmaSqrdInv);
            
            if(neighbor_ik.getVector() == x_j)//intentionally doing object equals check - should be same object
            {
                jIsNearBy = true;//yay, dont have to compute the distance ourselves
                numer = FastMath.exp(-(d_ik*d_ik) * sigmaSqrdInv);
            }
        }
        
        if(!jIsNearBy)
        {
            double d_ij = dm.dist(i, j, vecs, accelCache);
            numer = FastMath.exp(-(d_ij*d_ij) * sigmaSqrdInv);
        }
        
        return numer/(denom+1e-9);
    }
    
    private double p_ij(int i, int j, double sigma_i, double sigma_j, List<List<? extends VecPaired<Vec, Double>>> neighbors, List<Vec> vecs, List<Double> accelCache)
    {
        return (p_j_i(j, i, sigma_i, neighbors, vecs, accelCache)+p_j_i(i, j, sigma_j, neighbors, vecs, accelCache))/(2*neighbors.size());
    }
    
    /**
     * Computes the perplexity for the specified data point using the given sigma
     * @param i the data point to get the perplexity of
     * @param sigma the bandwidth to use
     * @param neighbors the set of nearest neighbors to consider
     * @return the perplexity 2<sup>H(P<sub>i</sub>)</sup>
     */
    private double perp(int i, int[][] nearMe, double sigma, List<List<? extends VecPaired<Vec, Double>>> neighbors, List<Vec> vecs, List<Double> accelCache)
    {
        //section 2 of Maaten, L. Van Der, & Hinton, G. (2008). Visualizing Data using t-SNE. Journal of Machine Learning Research, 9, 2579–2605.
        double hp = 0;

        for(int j_indx =0; j_indx < nearMe[i].length; j_indx++)
        {
            double p_ji = p_j_i(nearMe[i][j_indx], i, sigma, neighbors, vecs, accelCache);

            if (p_ji > 0) {
              hp += p_ji * FastMath.log2(p_ji);
            }
        }
        hp *= -1;
        
        return FastMath.pow2(hp);
    }
    
    
    private class Quadtree
    {
        public Node root;

        public Quadtree(double[] z )
        {
            this.root = new Node();
            this.root.minX = this.root.minY = Double.POSITIVE_INFINITY;
            this.root.maxX = this.root.maxY = -Double.POSITIVE_INFINITY;
            
            for(int i = 0; i < z.length/2; i++)
            {
                double x = z[i*2];
                double y = z[i*2+1];
                this.root.minX = Math.min(this.root.minX, x);
                this.root.maxX = Math.max(this.root.maxX, x);
                this.root.minY = Math.min(this.root.minY, y);
                this.root.maxY = Math.max(this.root.maxY, y);
            }
            
            //done b/c we have <= on min, so to get the edge we need to be slightly larger
            this.root.maxX = Math.nextUp(this.root.maxX);
            this.root.maxY = Math.nextUp(this.root.maxY);
            
            //nowe start inserting everything
            for(int i = 0; i < z.length/2; i++) {
              root.insert(1, i, z);
            }
        }
        
        
        
        private class Node implements Iterable<Node>
        {
            public int indx;
            public double x_mass, y_mass;
            public int N_cell;
            public double minX, maxX, minY, maxY;
            public Node NW, NE, SE, SW;

            public Node()
            {
                indx = -1;
                N_cell = 0;
                x_mass = y_mass = 0;
                NW = NE = SE = SW = null;
            }

            public Node(double minX, double maxX, double minY, double maxY)
            {
                this();
                this.minX = minX;
                this.maxX = maxX;
                this.minY = minY;
                this.maxY = maxY;
            }
            
            
            public boolean contains(int i, double[]z) 
            {
                double x = z[i*2];
                double y = z[i*2+1];
                
                return minX <= x && x < maxX && minY <= y && y < maxY;
            }
            
            public void insert(int weight, int i, double[] z)
            {
                x_mass += z[i*2];
                y_mass += z[i*2+1];
                N_cell+=weight;
                if(NW == null && indx < 0) {//was empy, just set
                  indx = i;
                } else
                {
                    if(indx >=0)
                    {
                        if(Math.abs(z[indx*2]- z[i*2]) < 1e-13 && 
                                Math.abs(z[indx*2+1]- z[i*2+1]) < 1e-13)
                        {
                            //near exact same value
                            //just let increase local weight indicate a "heavier" leaf
                            return;
                        }
                    }
                    if(NW == null)//we need to split
                    {
                        double w2 = (maxX-minX)/2;
                        double h2 = (maxY - minY)/2;

                        NW = new Node(minX,       minX + w2,  minY + h2,  maxY);
                        NE = new Node(minX + w2,  maxX,       minY + h2,  maxY);
                        SW = new Node(minX,       minX + w2,  minY,       minY + h2);
                        SE = new Node(minX + w2,  maxX,       minY,       minY + h2);

                        for(Node child : this) {
                          if(child.contains(this.indx, z))
                          {
                            child.insert(this.N_cell, this.indx, z);
                            break;
                          }
                    }
                        indx = -1;
                    }
                    //and pass this along to our children
                    for(Node child : this) {
                      if(child.contains(i, z))
                      {
                        child.insert(weight, i, z);
                        break;
                      }
                  }
                    
                }
            }

            public double diagLen()
            {
                double w = maxX-minX;
                double h = maxY-minY;
                return Math.sqrt(w*w+h*h);
            }
            
            @Override
            public Iterator<Node> iterator()
            {
                if(NW == null) {
                  return Collections.emptyIterator();
                } else {
                  return Arrays.asList(NW, NE, SW, SE).iterator();
                }
            }
            
        }
    }
}
