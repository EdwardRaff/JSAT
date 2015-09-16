package jsat.classifiers;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Edward Raff
 */
public class CategoricalData implements Cloneable, Serializable {

  private static final long serialVersionUID = 5783467611963064930L;

  public static CategoricalData[] copyOf(final CategoricalData[] orig) {
    final CategoricalData[] copy = new CategoricalData[orig.length];
    for (int i = 0; i < copy.length; i++) {
      copy[i] = orig[i].clone();
    }
    return copy;
  }

  private final int n;// Number of different categories
  private List<String> catNames;

  private String categoryName;

  /**
   *
   * @param n
   *          the number of categories
   */
  public CategoricalData(final int n) {
    this.n = n;
    catNames = new ArrayList<String>(n);
    for (int i = 0; i < n; i++) {
      catNames.add("Option " + (i + 1));
    }
    categoryName = "No Name";
  }

  @Override
  public CategoricalData clone() {
    final CategoricalData copy = new CategoricalData(n);

    if (catNames != null) {
      copy.catNames = new ArrayList<String>(catNames);
    }

    return copy;
  }

  public String getCategoryName() {
    return categoryName;
  }

  /**
   *
   * @return the number of possible categories there are for this category
   */
  public int getNumOfCategories() {
    return n;
  }

  public String getOptionName(final int i) {
    if (catNames != null) {
      return catNames.get(i);
    } else {
      return Integer.toString(i);
    }
  }

  public boolean isValidCategory(final int i) {
    return !(i < 0 || i >= n);
  }

  public void setCategoryName(final String categoryName) {
    this.categoryName = categoryName;
  }

  /**
   * Sets the name of one of the value options. Duplicate names are not allowed.
   * Trying to set the name of a non existent option will result in false being
   * returned. <br>
   * All names will be converted to lower case
   *
   * @param name
   *          the name to give
   * @param i
   *          the ith index to set.
   * @return true if the name was set. False if the name could not be set.
   */
  public boolean setOptionName(String name, final int i) {
    name = name.toLowerCase();
    if (i < 0 || i >= n) {
      return false;
    } else if (catNames.contains(name)) {
      return false;
    }
    catNames.set(i, name);

    return true;
  }

}
