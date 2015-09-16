package jsat.linear.vectorcollection.lsh;

import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.CosineDistanceNormalized;
import jsat.linear.vectorcollection.VectorArray;
import jsat.math.OnLineStatistics;
import jsat.utils.IntSet;

/**
 *
 * @author Edward Raff
 */
public class RandomProjectionLSHTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public RandomProjectionLSHTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of search method, of class RandomProjectionLSH.
   */
  @Test
  public void testSearch_Vec_double() {
    System.out.println("search");

    final List<VecPaired<Vec, Integer>> normalVecs = new ArrayList<VecPaired<Vec, Integer>>();
    final Random rand = new Random();

    for (int i = 0; i < 100; i++) {
      final DenseVector dv = new DenseVector(20);
      for (int j = 0; j < dv.length(); j++) {
        dv.set(j, rand.nextGaussian());
      }
      dv.normalize();
      normalVecs.add(new VecPaired<Vec, Integer>(dv, i));
    }

    final CosineDistanceNormalized dm = new CosineDistanceNormalized();

    final VectorArray<VecPaired<Vec, Integer>> naiveVC = new VectorArray<VecPaired<Vec, Integer>>(dm, normalVecs);
    final RandomProjectionLSH<VecPaired<Vec, Integer>> rpVC = new RandomProjectionLSH<VecPaired<Vec, Integer>>(
        normalVecs, 16, true);

    final OnLineStatistics knnStats = new OnLineStatistics();
    for (final Vec v : normalVecs) {
      knnStats.add(naiveVC.search(v, 11).get(10).getPair());// first nn is
                                                            // itselft
    }

    final double searchDist = knnStats.getMean() + knnStats.getStandardDeviation() * 2;
    final Set<Integer> inTruth = new IntSet();

    for (final Vec v : normalVecs)// now use the stats to compare results
    {
      final List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> trueResults = naiveVC.search(v, searchDist);
      final List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> aprxResults = rpVC.search(v, searchDist);

      inTruth.clear();
      for (final VecPaired<VecPaired<Vec, Integer>, Double> vp : trueResults) {
        inTruth.add(vp.getVector().getPair());
      }
      int contained = 0;
      for (final VecPaired<VecPaired<Vec, Integer>, Double> vp : aprxResults) {
        if (inTruth.contains(vp.getVector().getPair())) {
          contained++;
        }
      }

      // Recall must be at least 0.5, should be an easy target
      assertTrue(contained >= inTruth.size() / 2);
    }
  }

  /**
   * Test of search method, of class RandomProjectionLSH.
   */
  @Test
  public void testSearch_Vec_int() {
    System.out.println("search");
    final List<VecPaired<Vec, Integer>> normalVecs = new ArrayList<VecPaired<Vec, Integer>>();
    final Random rand = new Random();

    for (int i = 0; i < 100; i++) {
      final DenseVector dv = new DenseVector(20);
      for (int j = 0; j < dv.length(); j++) {
        dv.set(j, rand.nextGaussian());
      }
      dv.normalize();
      normalVecs.add(new VecPaired<Vec, Integer>(dv, i));
    }

    final CosineDistanceNormalized dm = new CosineDistanceNormalized();

    final VectorArray<VecPaired<Vec, Integer>> naiveVC = new VectorArray<VecPaired<Vec, Integer>>(dm, normalVecs);
    final RandomProjectionLSH<VecPaired<Vec, Integer>> rpVC = new RandomProjectionLSH<VecPaired<Vec, Integer>>(
        normalVecs, 16, true);

    final Set<Integer> inTruth = new IntSet();

    for (final Vec v : normalVecs)// now use the stats to compare results
    {
      final List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> trueResults = naiveVC.search(v, 15);
      final List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> aprxResults = rpVC.search(v, 15);

      inTruth.clear();
      for (final VecPaired<VecPaired<Vec, Integer>, Double> vp : trueResults) {
        inTruth.add(vp.getVector().getPair());
      }
      int contained = 0;
      for (final VecPaired<VecPaired<Vec, Integer>, Double> vp : aprxResults) {
        if (inTruth.contains(vp.getVector().getPair())) {
          contained++;
        }
      }

      // Recall must be at least 0.5, should be an easy target
      assertTrue(contained >= inTruth.size() / 2);
    }
  }

}
