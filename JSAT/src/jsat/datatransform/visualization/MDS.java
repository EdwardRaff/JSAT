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
package jsat.datatransform.visualization;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.datatransform.DataTransform;
import jsat.linear.DenseMatrix;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.math.OnLineStatistics;
import jsat.utils.FakeExecutor;
import jsat.utils.random.XORWOW;

/**
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class MDS {

  private static DistanceMetric embedMetric = new EuclideanDistance();

  private static double stress(final List<Vec> X_views, final List<Double> X_rowCache, final Matrix delta) {
    double stress = 0;

    for (int i = 0; i < delta.rows(); i++) {

      for (int j = i + 1; j < delta.rows(); j++) {
        final double tmp = embedMetric.dist(i, j, X_views, X_rowCache) - delta.get(i, j);
        stress += tmp * tmp;
      }
    }
    return stress;
  }

  private final DistanceMetric dm = new EuclideanDistance();
  private double tolerance = 1e-3;
  private final int maxIterations = 300;

  int targetSize = 2;

  public double getTolerance() {
    return tolerance;
  }

  public void setTolerance(final double tolerance) {
    this.tolerance = tolerance;
  }

  public <Type extends DataSet> Type transform(final DataSet<Type> d) {
    return transform(d, new FakeExecutor());
  }

  public <Type extends DataSet> Type transform(final DataSet<Type> d, final ExecutorService ex) {
    final List<Vec> orig_vecs = d.getDataVectors();
    final List<Double> orig_distCache = dm.getAccelerationCache(orig_vecs, ex);
    final int N = orig_vecs.size();

    // Delta is the true disimilarity matrix
    final Matrix delta = new DenseMatrix(N, N);

    final OnLineStatistics avg = new OnLineStatistics();
    for (int i = 0; i < d.getSampleSize(); i++) {
      for (int j = i + 1; j < d.getSampleSize(); j++) {
        final double dist = dm.dist(i, j, orig_vecs, orig_distCache);
        avg.add(dist);
        delta.set(i, j, dist);
        delta.set(j, i, dist);
      }
    }

    final Random rand = new XORWOW();

    final Matrix X = new DenseMatrix(N, targetSize);
    final List<Vec> X_views = new ArrayList<Vec>();
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < targetSize; j++) {
        X.set(i, j, rand.nextDouble());
      }
      X_views.add(X.getRowView(i));
    }
    List<Double> X_rowCache = embedMetric.getAccelerationCache(X_views, ex);

    // TODO, special case solution when all weights are the same, want to add
    // general case as well
    final Matrix V_inv = new DenseMatrix(N, N);
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        if (i == j) {
          V_inv.set(i, j, (1.0 - 1.0 / N) / N);
        } else {
          V_inv.set(i, j, (0.0 - 1.0 / N) / N);
        }
      }
    }

    double stressChange = Double.POSITIVE_INFINITY;
    double oldStress = stress(X_views, X_rowCache, delta);

    // the gutman transform matrix
    final Matrix B = new DenseMatrix(N, N);

    for (int iter = 0; iter < maxIterations && stressChange > tolerance; iter++) {

      // we need to set B correctly
      for (int i = 0; i < B.rows(); i++) {
        for (int j = i + 1; j < B.rows(); j++) {
          final double d_ij = embedMetric.dist(i, j, X_views, X_rowCache);

          if (d_ij > 1e-5) // avoid creating silly huge values
          {
            final double b_ij = -delta.get(i, j) / d_ij;// -w_ij if we support
                                                        // weights in the future
            B.set(i, j, b_ij);
            B.set(j, i, b_ij);
          } else {
            B.set(i, j, 0);
            B.set(j, i, 0);
          }
        }
      }
      // set the diagonal values
      for (int i = 0; i < B.rows(); i++) {
        B.set(i, i, 0);
        for (int k = 0; k < B.cols(); k++) {
          if (k != i) {
            B.increment(i, i, -B.get(i, k));
          }
        }
      }

      // Matrix X_new = V_inv.multiply(B, ex).multiply(X, ex);
      final Matrix X_new = B.multiply(X, ex);
      X_new.mutableMultiply(1.0 / N);

      X_new.copyTo(X);
      X_rowCache = embedMetric.getAccelerationCache(X_views, ex);

      final double newStress = stress(X_views, X_rowCache, delta);
      stressChange = Math.abs(oldStress - newStress);
      oldStress = newStress;
    }

    final DataSet<Type> transformed = d.shallowClone();

    final IdentityHashMap<DataPoint, Integer> indexMap = new IdentityHashMap<DataPoint, Integer>(N);
    for (int i = 0; i < N; i++) {
      indexMap.put(d.getDataPoint(i), i);
    }

    transformed.applyTransform(new DataTransform() {

      /**
       *
       */
      private static final long serialVersionUID = 1L;

      @Override
      public DataTransform clone() {
        return this;
      }

      @Override
      public DataPoint transform(final DataPoint dp) {
        final int i = indexMap.get(dp);

        return new DataPoint(X.getRow(i), dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
      }
    });

    return (Type) transformed;
  }
}
