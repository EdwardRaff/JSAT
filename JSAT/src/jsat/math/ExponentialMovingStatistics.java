/*
 * Copyright (C) 2016 Edward Raff <Raff.Edward@gmail.com>
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
package jsat.math;

import java.io.Serializable;

/**
 * This class keeps track of a set of Exponential Moving statistics (the mean
 * and standard deviation). When considering just the mean, this is often
 * referred to as Exponential Moving Average (EMA). Similar to
 * {@link OnLineStatistics}, this method will use fixed memory to keep an
 * estimate of the mean and standard deviation of a stream of values. However
 * this class will adjust to the mean and standard deviation of only recent
 * additions, and will "forget" the contribution of earlier values. The rate of
 * forgetting is controlled with the {@link #smoothing smoothing} parameter.
 *
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class ExponentialMovingStatistics implements Serializable, Cloneable
{
    private double mean;
    private double variance;
    private double smoothing;
    
    /**
     * Creates a new object for keeping an exponential estimate of the mean and
     * variance. Uses a relatively low smoothing factor of 0.1
     *
     */
    public ExponentialMovingStatistics()
    {
        this(0.1);
    }

    /**
     * Creates a new object for keeping an exponential estimate of the mean and
     * variance
     *
     * @param smoothing the {@link #smoothing smoothing} parameter to use
     */
    public ExponentialMovingStatistics(double smoothing)
    {
        this(smoothing, Double.NaN, 0);
    }

    /**
     * Creates a new object for keeping an exponential estimate of the mean and
     * variance
     *
     * @param smoothing the {@link #smoothing smoothing} parameter to use
     * @param mean an initial mean. May be {@link Double#NaN NaN} to indicate no
     * initial mean.
     * @param variance an initial variance. May be {@link Double#NaN NaN} to
     * indicate no initial mean.
     */
    public ExponentialMovingStatistics(double smoothing, double mean, double variance)
    {
        this.mean = mean;
        this.variance = variance;
        setSmoothing(smoothing);
    }

    /**
     * Sets the smoothing parameter value to use. Must be in the range (0, 1].
     * Changing this value will impact how quickly the statistics adapt to
     * changes, with larger values increasing rate of change and smaller values
     * decreasing it.
     *
     * @param smoothing the smoothing value to use
     */
    public void setSmoothing(double smoothing)
    {
        if (smoothing <= 0 || smoothing > 1 || Double.isNaN(smoothing))
            throw new IllegalArgumentException("Smoothing must be in (0, 1], not " + smoothing);
        this.smoothing = smoothing;
    }

    /**
     *
     * @return the smoothing parameter in use
     */
    public double getSmoothing()
    {
        return smoothing;
    }

    /**
     * Adds the given data point to the statistics
     *
     * @param x the new value to add to the moving statistics
     */
    public void add(double x)
    {
        if (Double.isNaN(mean))//fist case
        {
            mean = x;
            variance = 0;
        }
        else//general case
        {
            //first update stnd deviation 
            variance = (1-smoothing)*(variance + smoothing*Math.pow(x-mean, 2));
            mean = (1-smoothing)*mean + smoothing*x;
        }
                    
    }

    /**
     * 
     * @return estimate of the moving mean
     */
    public double getMean()
    {
        return mean;
    }

    /**
     * 
     * @return the estimate of moving variance
     */
    public double getVariance()
    {
        return variance;
    }

    /**
     * 
     * @return the estimate of moving standard deviation
     */
    public double getStandardDeviation()
    {
        return Math.sqrt(getVariance()+1e-13);
    }

}
