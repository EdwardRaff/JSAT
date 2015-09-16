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
package jsat.distributions.discrete;

/**
 * The discrete uniform distribution.
 *
 * @author Edward Raff
 */
public class UniformDiscrete extends DiscreteDistribution {

  /**
   *
   */
  private static final long serialVersionUID = 1L;
  private int min;
  private int max;

  /**
   * Creates a new Uniform distribution with a min of 0 and a max of 10
   */
  public UniformDiscrete() {
    this(0, 10);
  }

  /**
   * Creates a new discrete uniform distribution
   *
   * @param min
   *          the minimum value to occur
   * @param max
   *          the maximum value to occur
   */
  public UniformDiscrete(final int min, final int max) {
    setMinMax(min, max);
  }

  @Override
  public double cdf(final int x) {
    if (x >= max) {
      return 1;
    } else if (x < min) {
      return 0;
    } else {
      return (1 - min + x) / (double) (1 + max - min);
    }
  }

  @Override
  public DiscreteDistribution clone() {
    return new UniformDiscrete(min, max);
  }

  public int getMax() {
    return max;
  }

  public int getMin() {
    return min;
  }

  @Override
  public double invCdf(final double p) {
    if (p <= 0) {
      return min;
    } else if (p >= 1) {
      return max;
    } else {
      return Math.max(1, Math.ceil((1 + max - min) * p) + min - 1);
    }
  }

  @Override
  public double max() {
    return max;
  }

  @Override
  public double mean() {
    return max / 2.0 + min / 2.0;
  }

  @Override
  public double median() {
    return Math.floor(mean());
  }

  @Override
  public double min() {
    return min;
  }

  @Override
  public double mode() {
    return Double.NaN;
  }

  @Override
  public double pmf(final int x) {
    if (x < min || x > max) {
      return 0;
    } else {
      return 1.0 / (1 + max - min);
    }
  }

  /**
   * Sets the maximum value to occur from the distribution, must be greater than
   * {@link #getMin() }.
   *
   * @param max
   *          the maximum value to occur
   */
  public void setMax(final int max) {
    if (max <= min) {
      throw new IllegalArgumentException(max + " must be greater than the min value " + min);
    }
    this.max = max;
  }

  /**
   * Sets the minimum value to occur from the distribution, must be less than
   * {@link #getMax() }.
   *
   * @param min
   *          the minimum value to occur
   */
  public void setMin(final int min) {
    if (min >= max) {
      throw new IllegalArgumentException(min + " must be less than the max value " + max);
    }
    this.min = min;
  }

  /**
   * Sets the minimum and maximum values at the same time, this is useful if
   * setting them one at a time may have caused a conflict with the previous
   * values
   *
   * @param min
   *          the new minimum value to occur
   * @param max
   *          the new maximum value to occur
   */
  public void setMinMax(final int min, final int max) {
    if (min >= max) {
      throw new IllegalArgumentException(
          "The input minimum (" + min + ") must be less than the given max (" + max + ")");
    }
    this.min = min;
    this.max = max;
  }

  @Override
  public double skewness() {
    return 0;
  }

  @Override
  public double variance() {
    long dif = max - min + 1;
    dif *= dif;
    return (dif - 1) / 12.0;
  }

}
