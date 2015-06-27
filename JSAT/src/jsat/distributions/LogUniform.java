/*
 * Copyright (C) 2015 Edward Raff <Raff.Edward@gmail.com>
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
package jsat.distributions;

import jsat.linear.Vec;

/**
 * The Log Uniform distribution is such that if X is the distribution, then Y =
 * log(X) is uniformly distributed. Because of this log term, this distribution
 * can only take values in a positive range.
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class LogUniform extends ContinuousDistribution
{
    private double min, max;
    private double logMin, logMax;
    private double logDiff;
    private double diff;

    /**
     * Creates a new Log Uniform distribution between 1e-2 and 1
     */
    public LogUniform()
    {
        this(1e-2, 1);
    }

    /**
     * Creates a new Log Uniform distribution
     * 
     * @param min the minimum value to be returned by this distribution 
     * @param max the maximum value to be returned by this distribution
     */
    public LogUniform(double min, double max)
    {
        setMinMax(min, max);
    }

    /**
     * Sets the minimum and maximum values for this distribution 
     * @param min the minimum value, must be positive
     * @param max the maximum value, must be larger than {@code min}
     */
    public void setMinMax(double min, double max)
    {
        if(min <= 0 || Double.isNaN(min) || Double.isInfinite(min))
            throw new IllegalArgumentException("min value must be positive, not " + min);
        else if(min >= max || Double.isNaN(max) || Double.isInfinite(max))
            throw new IllegalArgumentException("max (" + max + ") must be larger than min (" + min+")" );
        this.max = max;
        this.min = min;
        this.logMax = Math.log(max);
        this.logMin = Math.log(min);
        this.logDiff = logMax-logMin;
        this.diff = max-min;
    }
    
    @Override
    public double pdf(double x)
    {
        if(x < min)
            return 0;
        else if(x > max)
            return 0;
        else
            return 1.0/(x*(logMax-logMin));
    }

    @Override
    public String getDistributionName()
    {
        return "LogUniform";
    }

    @Override
    public String[] getVariables()
    {
        return new String[]{"min", "max"};
    }

    @Override
    public double[] getCurrentVariableValues()
    {
        return new double[]{min, max};
    }

    @Override
    public void setVariable(String var, double value)
    {
        if(var.equals("min"))
            setMinMax(value, max);
        else if(var.equals("max"))
            setMinMax(min, value);
    }

    @Override
    public LogUniform clone()
    {
        return new LogUniform(min, max);
    }

    @Override
    public void setUsingData(Vec data)
    {
        //probably could do way better, but whatever
        double guessMin = data.min();
        double guessMax = data.max();
        setMinMax(Math.max(guessMin, 1e-10), guessMax);
    }

    @Override
    public double cdf(double x)
    {
        if(x < min)
            return 0;
        else if(x > max)
            return 1;
        else
            return (Math.log(x)-logMin)/(logDiff);
    }

    @Override
    public double invCdf(double p)
    {
        if(p < 0 || p > 1 || Double.isNaN(p))
            throw new IllegalArgumentException("p must be in [0,1], not " + p);
        return Math.exp(p*logMax-p*logMin)*min;
    }

    @Override
    public double mean()
    {
        return (diff)/(logDiff);
    }

    @Override
    public double median()
    {
        return Math.sqrt(min)*Math.sqrt(max);
    }

    @Override
    public double mode()
    {
        return min();
    }

    @Override
    public double variance()
    {
        return (max*max-min*min)/(2*logDiff) - diff*diff/(logDiff*logDiff);
    }

    @Override
    public double skewness()
    {
        return Double.NaN;//TODO derive 
    }

    @Override
    public double min()
    {
        return min;
    }

    @Override
    public double max()
    {
        return max;
    }
    
}
