/*
 * Copyright (C) 2019 Edward Raff
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
package jsat.clustering.evaluation;

import java.util.List;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;

/**
 * ﻿A clustering result satisfies homogeneity if all of its clusters contain
 * only data points which are members of a single class. Normally, Homogeneity
 * would return a score with 0 being undesirable, and 1 being desirable.
 *
 * @author Edward Raff
 */
public class Homogeneity implements ClusterEvaluation
{

    public Homogeneity()
    {
    }

    @Override
    public double evaluate(int[] designations, DataSet dataSet)
    {
	if( !(dataSet instanceof ClassificationDataSet))
            throw new RuntimeException("Homogeneity can only be calcuate for classification data sets");
        ClassificationDataSet cds = (ClassificationDataSet)dataSet;
        int clusters = 0;//how many clusters are there? 
        for(int clusterID : designations)
            clusters = Math.max(clusterID+1, clusters);
	int C = cds.getPredicting().getNumOfCategories();
	int K = clusters;
	
	double[][] A = new double[C][K];
	//Rows of AK
	double[] class_sum = new double[C];
	double[] cluster_sum = new double[K];
	double n = 0.0;
        for(int i = 0; i < designations.length; i++)
        {
            int cluster = designations[i];
            if(cluster < 0)
                continue;//noisy point 
            int label = cds.getDataPointCategory(i);
            double weight = cds.getWeight(i);
            A[label][cluster] += weight;
            class_sum[label] += weight;
            cluster_sum[cluster] += weight;
            n += weight;
        }
	
	double h_ck = 0;
	double h_c = 0;
	
	for(int c = 0; c < C; c++)
	{
	    //compute h_c, h_c needs only one loop to compute its values
	    if(class_sum[c] > 0)
		h_c -= class_sum[c]/n * (Math.log(class_sum[c])-Math.log(n));
	    
	    for(int k = 0; k < K; k++)
	    {
		//h(c|k) calc
		if(A[c][k] == 0 || cluster_sum[k] == 0)
		    continue;
		h_ck -= A[c][k]/n * (Math.log(A[c][k]) - Math.log(cluster_sum[k]));
	    }
	}
	
	if(h_c == 0)
	    return 0;
	return h_ck/h_c;
    }

    @Override
    public double evaluate(List<List<DataPoint>> dataSets)
    {
	throw new UnsupportedOperationException("Homogeneity requires the true data set"
                + " labels, call evaluate(int[] designations, DataSet dataSet)"
                + " instead");
    }

    @Override
    public double naturalScore(double evaluate_score)
    {
	return 1-evaluate_score;
    }

    @Override
    public Homogeneity clone()
    {
	return new Homogeneity();
    }
    
}
