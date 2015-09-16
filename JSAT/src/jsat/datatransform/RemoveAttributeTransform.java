package jsat.datatransform;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.utils.IntList;
import jsat.utils.IntSet;

/**
 * This Data Transform allows the complete removal of specific features from the
 * data set.
 *
 * @author Edward Raff
 */
public class RemoveAttributeTransform implements DataTransform {

  /**
   * Factory for producing {@link RemoveAttributeTransform} transforms
   */
  public static class RemoveAttributeTransformFactory implements DataTransformFactory {

    private final Set<Integer> catToRemove;
    private final Set<Integer> numerToRemove;

    public RemoveAttributeTransformFactory(final Set<Integer> catToRemove, final Set<Integer> numerToRemove) {
      this.catToRemove = catToRemove;
      this.numerToRemove = numerToRemove;
    }

    @Override
    public RemoveAttributeTransformFactory clone() {
      return new RemoveAttributeTransformFactory(new IntSet(catToRemove), new IntSet(numerToRemove));
    }

    @Override
    public DataTransform getTransform(final DataSet dataset) {
      return new RemoveAttributeTransform(dataset, catToRemove, numerToRemove);
    }
  }

  private static final long serialVersionUID = 8803223213862922734L;
  /*
   * Each index map maps the old indecies in the original data set to their new
   * positions. The value in the array is old index, the index of the value is
   * the index it would be when the attributes were removed. This means each is
   * in sorted order, and is of the size of the resulting feature space
   */
  protected int[] catIndexMap;

  protected int[] numIndexMap;

  /**
   * Empty constructor that may be used by extending classes. Transforms that
   * extend this will need to call null
   * {@link #setUp(jsat.DataSet, java.util.Set, java.util.Set) } once the
   * attributes to remove have been selected
   */
  protected RemoveAttributeTransform() {

  }

  /**
   * Creates a new transform for removing specified features from a data set
   *
   * @param dataSet
   *          the data set that this transform is meant for
   * @param categoricalToRemove
   *          the set of categorical attributes to remove, in the rage of [0,
   *          {@link DataSet#getNumCategoricalVars() }).
   * @param numericalToRemove
   *          the set of numerical attributes to remove, in the rage of [0,
   *          {@link DataSet#getNumNumericalVars() }).
   */
  public RemoveAttributeTransform(final DataSet dataSet, final Set<Integer> categoricalToRemove,
      final Set<Integer> numericalToRemove) {
    setUp(dataSet, categoricalToRemove, numericalToRemove);
  }

  /**
   * Copy constructor
   *
   * @param other
   *          the transform to copy
   */
  protected RemoveAttributeTransform(final RemoveAttributeTransform other) {
    catIndexMap = Arrays.copyOf(other.catIndexMap, other.catIndexMap.length);
    numIndexMap = Arrays.copyOf(other.numIndexMap, other.numIndexMap.length);
  }

  @Override
  public RemoveAttributeTransform clone() {
    return new RemoveAttributeTransform(this);
  }

  /**
   * A serious of Remove Attribute Transforms may be learned and applied
   * sequentially to a single data set. Instead of keeping all the transforms
   * around indefinitely, a sequential series of Remove Attribute Transforms can
   * be consolidated into a single transform object. <br>
   * This method mutates the this transform by providing it with the transform
   * that would have been applied before this current object. Once complete,
   * this transform can be used two perform both removals in one step.<br>
   * <br>
   * Example: <br>
   * An initial set of features <i>A</i> is transformed into <i>A'</i> by
   * transform t<sub>1</sub><br>
   * <i>A'</i> is transformed into <i>A''</i> by transform t<sub>2</sub><br>
   * Instead, you can invoke t<sub>2</sub>.consolidate(t<sub>1</sub>). You can
   * then transform <i>A</i> into <i>A''</i> by using only transform t
   * <sub>2</sub>
   *
   *
   * @param preceding
   *          the DataTransform that immediately precedes this one in a
   *          sequential list of transforms
   */
  public void consolidate(final RemoveAttributeTransform preceding) {
    for (int i = 0; i < catIndexMap.length; i++) {
      catIndexMap[i] = preceding.catIndexMap[catIndexMap[i]];
    }
    for (int i = 0; i < numIndexMap.length; i++) {
      numIndexMap[i] = preceding.numIndexMap[numIndexMap[i]];
    }
  }

  /**
   * Returns an unmodifiable list of the original indices of the nominal
   * attributes that will be kept when this transform is applied.
   *
   * @return the nominal indices that are not removed by this transform
   */
  public List<Integer> getKeptNominal() {
    return IntList.unmodifiableView(catIndexMap, catIndexMap.length);
  }

  /**
   * Returns an unmodifiable list of the original indices of the numeric
   * attributes that will be kept when this transform is applied.
   *
   * @return the numeric indices that are not removed by this transform
   */
  public List<Integer> getKeptNumeric() {
    return IntList.unmodifiableView(numIndexMap, numIndexMap.length);
  }

  /**
   * Returns a mapping from the nominal indices in the transformed space back to
   * their original indices
   *
   * @return a mapping from the transformed nominal space to the original one
   */
  public Map<Integer, Integer> getReverseNominalMap() {
    final Map<Integer, Integer> map = new HashMap<Integer, Integer>();
    for (int newIndex = 0; newIndex < catIndexMap.length; newIndex++) {
      map.put(newIndex, catIndexMap[newIndex]);
    }
    return map;
  }

  /**
   * Returns a mapping from the numeric indices in the transformed space back to
   * their original indices
   *
   * @return a mapping from the transformed numeric space to the original one
   */
  public Map<Integer, Integer> getReverseNumericMap() {
    final Map<Integer, Integer> map = new HashMap<Integer, Integer>();
    for (int newIndex = 0; newIndex < numIndexMap.length; newIndex++) {
      map.put(newIndex, numIndexMap[newIndex]);
    }
    return map;
  }

  /**
   * Sets up the Remove Attribute Transform properly
   *
   * @param dataSet
   *          the data set to remove the attributes from
   * @param categoricalToRemove
   *          the categorical attributes to remove
   * @param numericalToRemove
   *          the numeric attributes to remove
   */
  protected final void setUp(final DataSet dataSet, final Set<Integer> categoricalToRemove,
      final Set<Integer> numericalToRemove) {
    for (final int i : categoricalToRemove) {
      if (i >= dataSet.getNumCategoricalVars()) {
        throw new RuntimeException("The data set does not have a categorical value " + i + " to remove");
      }
    }
    for (final int i : numericalToRemove) {
      if (i >= dataSet.getNumNumericalVars()) {
        throw new RuntimeException("The data set does not have a numercal value " + i + " to remove");
      }
    }

    catIndexMap = new int[dataSet.getNumCategoricalVars() - categoricalToRemove.size()];
    numIndexMap = new int[dataSet.getNumNumericalVars() - numericalToRemove.size()];
    int k = 0;
    for (int i = 0; i < dataSet.getNumCategoricalVars(); i++) {
      if (categoricalToRemove.contains(i)) {
        continue;
      }
      catIndexMap[k++] = i;
    }
    k = 0;
    for (int i = 0; i < dataSet.getNumNumericalVars(); i++) {
      if (numericalToRemove.contains(i)) {
        continue;
      }
      numIndexMap[k++] = i;
    }
  }

  @Override
  public DataPoint transform(final DataPoint dp) {
    final int[] catVals = dp.getCategoricalValues();
    final Vec numVals = dp.getNumericalValues();

    final CategoricalData[] newCatData = new CategoricalData[catIndexMap.length];
    final int[] newCatVals = new int[newCatData.length];
    Vec newNumVals;
    if (numVals.isSparse()) {
      if (numVals instanceof SparseVector) {
        newNumVals = new SparseVector(numIndexMap.length, numVals.nnz());
      } else {
        newNumVals = new SparseVector(numIndexMap.length);
      }
    } else {
      newNumVals = new DenseVector(numIndexMap.length);
    }

    for (int i = 0; i < catIndexMap.length; i++) {
      newCatVals[i] = catVals[catIndexMap[i]];
    }

    final int k = 0;

    final Iterator<IndexValue> iter = numVals.getNonZeroIterator();
    if (iter.hasNext()) // if all values are zero, nothing to do
    {
      IndexValue curIV = iter.next();
      for (int i = 0; i < numIndexMap.length; i++)// i is the old index
      {
        if (numVals.isSparse()) // log(n) insert and loopups to avoid!
        {
          if (curIV == null) {
            continue;
          }
          if (numIndexMap[i] > curIV.getIndex()) {// We skipped a value that
                                                  // existed
            while (numIndexMap[i] > curIV.getIndex() && iter.hasNext()) {
              curIV = iter.next();
            }
          }
          if (numIndexMap[i] < curIV.getIndex()) {
            // Index is zero, nothing to set
            // Index is zero, nothing to set
          } else if (numIndexMap[i] == curIV.getIndex()) {
            newNumVals.set(i, curIV.getValue());
            if (iter.hasNext()) {
              curIV = iter.next();
            } else {
              curIV = null;
            }
          }
        } else {
          // All dense, just set them all
          newNumVals.set(i, numVals.get(numIndexMap[i]));
        }
      }
    }
    return new DataPoint(newNumVals, newCatVals, newCatData, dp.getWeight());
  }
}
