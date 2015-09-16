package jsat.linear.vectorcollection.lsh;

import static org.junit.Assert.assertEquals;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.EuclideanDistance;

/**
 *
 * @author Edward Raff
 */
public class E2LSHTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public E2LSHTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of searchR method, of class E2LSH.
   */
  @Test
  public void testSearchR_Vec() {
    System.out.println("searchR");
    // Test case, create an easy data set of 10 clusters, get all the neighbors
    // for each main cluster
    // use dimension as the number of clusters and how many poitns are in each
    // cluster
    final int dim = 10;
    final List<Vec> mainVecs = new ArrayList<Vec>(dim);
    for (int i = 0; i < dim; i++) {
      final DenseVector dv = new DenseVector(dim);
      dv.set(i, dim * dim);
      mainVecs.add(dv);
    }

    final Random rand = new Random();
    final List<Vec> extraVecs = new ArrayList<Vec>();
    for (int i = 0; i < mainVecs.size(); i++) {
      for (final Vec mainVec : mainVecs) {
        final DenseVector newVec = new DenseVector(dim);
        newVec.set(i, dim * dim);
        for (int k = 0; k < newVec.length(); k++) {
          newVec.increment(k, rand.nextDouble());
        }
        extraVecs.add(newVec);
      }
    }

    final List<Vec> allVecs = new ArrayList<Vec>(mainVecs);
    allVecs.addAll(extraVecs);
    Collections.shuffle(allVecs);

    final E2LSH<Vec> e2lsh = new E2LSH<Vec>(allVecs, dim, 0.1, 4, 5, 0.1, new EuclideanDistance());

    for (final Vec v : mainVecs) {
      assertEquals(dim + 1, e2lsh.searchR(v).size());
    }
  }

}
