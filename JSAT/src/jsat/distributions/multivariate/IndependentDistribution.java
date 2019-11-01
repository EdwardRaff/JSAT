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
package jsat.distributions.multivariate;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import jsat.distributions.ContinuousDistribution;
import jsat.distributions.Distribution;
import jsat.distributions.discrete.DiscreteDistribution;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class IndependentDistribution implements MultivariateDistribution
{
    protected List<Distribution> distributions;

    public IndependentDistribution(List<Distribution> distributions) 
    {
        this.distributions = distributions;
    }
    
    public IndependentDistribution(IndependentDistribution toCopy) 
    {
        this.distributions = toCopy.distributions.stream()
                .map(Distribution::clone)
                .collect(Collectors.toList());
    }
    
    @Override
    public double logPdf(Vec x) 
    {
        if(x.length() != distributions.size())
            throw new ArithmeticException("Expected input of size " + distributions.size() + " not " + x.length());
        double logPDF = 0;
        for(int i = 0; i < x.length(); i++)
        {
            Distribution dist = distributions.get(i);
            if(dist instanceof DiscreteDistribution)
                logPDF += ((DiscreteDistribution)dist).logPmf((int) Math.round(x.get(i)));
            else
                logPDF += ((ContinuousDistribution) dist).logPdf(x.get(i));   
        }
        
        return logPDF;
    }

    @Override
    public <V extends Vec> boolean setUsingData(List<V> dataSet, boolean parallel) 
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public MultivariateDistribution clone() 
    {
        return new IndependentDistribution(this);
    }

    @Override
    public List<Vec> sample(int count, Random rand) 
    {
        List<Vec> sample = new ArrayList<>();
        for(int i = 0; i < count; i++)
        {
            sample.add(new DenseVector(distributions.size()));
        }

        for (int j = 0; j < distributions.size(); j++) 
        {
            Distribution d = distributions.get(j);
            double[] vals = d.sample(count, rand);
            for(int i = 0; i < sample.size(); i++)
                sample.get(i).set(j, vals[i]);
        }
        
        return sample;
    }

}
