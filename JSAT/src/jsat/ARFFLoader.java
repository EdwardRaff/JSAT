package jsat;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.Reader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.regression.RegressionDataSet;

/**
 * Class for loading ARFF files. ARFF is a human readable file format used by
 * Weka. The ARFF file formal allows for attributes that have missing
 * information, which is not supported by JSAT. Any data point with missing
 * information will be skipped in the loading process.
 *
 * <br>
 * <a href="http://www.cs.waikato.ac.nz/ml/weka/arff.html">About Weka</a>
 *
 * @author Edward Raff
 */
public class ARFFLoader {

  private static String addQuotes(final String string) {
    if (string.contains(" ")) {
      return "\"" + string + "\"";
    } else {
      return string;
    }
  }

  /**
   * Uses the given file path to load a data set from an ARFF file.
   *
   * @param file
   *          the path to the ARFF file to load
   * @return the data set from the ARFF file, or null if the file could not be
   *         loaded.
   */
  public static SimpleDataSet loadArffFile(final File file) {
    try {
      return loadArffFile(new FileReader(file));
    } catch (final FileNotFoundException ex) {
      Logger.getLogger(ARFFLoader.class.getName()).log(Level.SEVERE, null, ex);
      return null;
    }
  }

  /**
   * Uses the given reader to load a data set assuming it follows the ARFF file
   * format
   *
   * @param input
   *          the reader to load the data set from
   * @return the data set from the stream, or null of the file could not be
   *         loaded
   */
  public static SimpleDataSet loadArffFile(final Reader input) {
    final ArrayList<DataPoint> list = new ArrayList<DataPoint>();

    final BufferedReader br = new BufferedReader(input);

    int numOfVars = 0;
    int numReal = 0;
    final List<Boolean> isReal = new ArrayList<Boolean>();
    final List<String> variableNames = new ArrayList<String>();
    final List<HashMap<String, Integer>> catVals = new ArrayList<HashMap<String, Integer>>();
    String line = null;
    CategoricalData[] categoricalData = null;
    try {
      boolean atData = false;
      while ((line = br.readLine()) != null) {
        if (line.startsWith("%") || line.trim().isEmpty()) {
          continue;/// Its a comment, skip
        }

        line = line.trim();

        if (line.startsWith("@") && !atData) {
          line = line.substring(1).toLowerCase();

          if (line.toLowerCase().startsWith("data")) {
            categoricalData = new CategoricalData[numOfVars - numReal];

            int k = 0;
            for (int i = 0; i < catVals.size(); i++) {
              if (catVals.get(i) != null) {
                categoricalData[k] = new CategoricalData(catVals.get(i).size());
                categoricalData[k].setCategoryName(variableNames.get(i));
                for (final Entry<String, Integer> entry : catVals.get(i).entrySet()) {
                  categoricalData[k].setOptionName(entry.getKey(), entry.getValue());
                }
                k++;
              }
            }

            atData = true;
            continue;
          } else if (!line.toLowerCase().startsWith("attribute")) {
            continue;
          }
          numOfVars++;
          line = line.substring("attribute".length()).trim();// Remove the
                                                             // space, it could
                                                             // be multiple
                                                             // spaces

          String variableName = null;
          line = line.replace("\t", " ");
          if (line.startsWith("'")) {
            final Pattern p = Pattern.compile("'.+?'");
            final Matcher m = p.matcher(line);
            m.find();
            variableName = nameTrim(m.group());

            line = line.replaceFirst("'.+?'", "placeHolder");
          } else {
            variableName = nameTrim(line.trim().replaceAll("\\s+.*", ""));
          }
          variableNames.add(variableName);
          final String[] tmp = line.split("\\s+", 2);

          if (tmp[1].trim().equals("real") || tmp[1].trim().equals("numeric") || tmp[1].trim().startsWith("integer")) {
            numReal++;
            isReal.add(true);
            catVals.add(null);
          } else// Not correct, but we arent supporting anything other than real
                // and categorical right now
          {
            isReal.add(false);
            String cats = tmp[1].replace("{", "").replace("}", "").trim();
            if (cats.endsWith(",")) {
              cats = cats.substring(0, cats.length() - 1);
            }
            final String[] catValsRaw = cats.split(",");
            final HashMap<String, Integer> tempMap = new HashMap<String, Integer>();
            for (int i = 0; i < catValsRaw.length; i++) {
              catValsRaw[i] = nameTrim(catValsRaw[i]);
              tempMap.put(catValsRaw[i], i);
            }
            catVals.add(tempMap);
          }
        } else if (atData && !line.isEmpty()) {
          if (line.contains("?")) {// We dont handle missing data
            continue;
          }
          double weight = 1.0;
          final String[] tmp = line.split(",");
          if (tmp.length != isReal.size()) {
            final String s = tmp[isReal.size()];
            if (tmp.length == isReal.size() + 1) // {#} means the # is the
                                                 // weight
            {
              if (!s.matches("\\{\\d+(\\.\\d+)?\\}")) {
                throw new RuntimeException(
                    "extra column must indicate a data point weigh in the form of \"{#}\", instead bad token " + s
                        + " was found");
              }
              weight = Double.parseDouble(s.substring(1, s.length() - 1));
            } else {
              throw new RuntimeException("Column had " + tmp.length + " values instead of " + isReal.size());
            }
          }

          final DenseVector vec = new DenseVector(numReal);

          final int[] cats = new int[numOfVars - numReal];
          int k = 0;// Keeping track of position in cats
          for (int i = 0; i < isReal.size(); i++) {
            if (isReal.get(i)) {
              vec.set(i - k, Double.parseDouble(tmp[i].trim()));
            } else// Categorical
            {
              tmp[i] = nameTrim(tmp[i]);
              cats[k++] = catVals.get(i).get(tmp[i].trim().toLowerCase());
            }
          }

          list.add(new DataPoint(vec, cats, categoricalData, weight));
        }
      }
    } catch (final IOException ex) {

    }

    final SimpleDataSet dataSet = new SimpleDataSet(list);
    int k = 0;
    for (int i = 0; i < isReal.size(); i++) {
      if (isReal.get(i)) {
        dataSet.setNumericName(variableNames.get(k), k++);
      }
    }

    return dataSet;
  }

  /**
   * Removes the quotes at the end and front of a string if there are any, as
   * well as spaces at the front and end
   *
   * @param in
   *          the string to trim
   * @return the white space and quote trimmed string
   */
  private static String nameTrim(String in) {
    in = in.trim();
    if (in.startsWith("'") || in.startsWith("\"")) {
      in = in.substring(1);
    }
    if (in.endsWith("'") || in.startsWith("\"")) {
      in = in.substring(0, in.length() - 1);
    }
    return in.trim();
  }

  public static void writeArffFile(final DataSet data, final OutputStream os) {
    writeArffFile(data, os, "Default_Relation");
  }

  /**
   * Writes out the dataset as an ARFF file to the given stream. This method
   * will automatically handle the target variable of
   * {@link ClassificationDataSet} and {@link RegressionDataSet}.
   *
   * @param data
   *          the dataset to write out
   * @param os
   *          the output stream to write too
   * @param relation
   *          the relation label to write out
   */
  public static void writeArffFile(final DataSet data, final OutputStream os, final String relation) {
    final PrintWriter writer = new PrintWriter(os);
    // write out the relation tag
    writer.write(String.format("@relation %s\n", addQuotes(relation)));

    // write out attributes
    // first all categorical features
    final CategoricalData[] catInfo = data.getCategories();
    for (final CategoricalData cate : catInfo) {
      writeCatVar(writer, cate);
    }
    // write out all numeric features
    for (int i = 0; i < data.getNumNumericalVars(); i++) {
      final String name = data.getNumericName(i);
      writer.write("@attribute " + (name == null ? "num" + i : name.replaceAll("\\s+", "-")) + " NUMERIC\n");
    }
    if (data instanceof ClassificationDataSet) {// also write out class variable
      writeCatVar(writer, ((ClassificationDataSet) data).getPredicting());
    }
    if (data instanceof RegressionDataSet) {
      writer.write("@ATTRIBUTE target NUMERIC\n");
    }
    writer.write("@DATA\n");
    for (int row = 0; row < data.getSampleSize(); row++) {
      final DataPoint dp = data.getDataPoint(row);
      boolean firstFeature = true;
      // cat vars first
      for (int i = 0; i < catInfo.length; i++) {
        if (!firstFeature) {
          writer.write(",");
        }
        firstFeature = false;
        writer.write(addQuotes(catInfo[i].getOptionName(dp.getCategoricalValue(i))));
      }
      // numeric vars
      final Vec v = dp.getNumericalValues();
      for (int i = 0; i < v.length(); i++) {
        if (!firstFeature) {
          writer.write(",");
        }
        firstFeature = false;
        writer.write(Double.toString(v.get(i)));
      }
      if (data instanceof ClassificationDataSet) // also write out class
                                                 // variable
      {
        if (!firstFeature) {
          writer.write(",");
        }
        firstFeature = false;
        final ClassificationDataSet cdata = (ClassificationDataSet) data;
        writer.write(addQuotes(cdata.getPredicting().getOptionName(cdata.getDataPointCategory(row))));
      }
      if (data instanceof RegressionDataSet) {
        if (!firstFeature) {
          writer.write(",");
        }
        firstFeature = false;
        writer.write(Double.toString(((RegressionDataSet) data).getTargetValue(row)));
      }
      writer.write("\n");
    }
    writer.flush();
  }

  private static void writeCatVar(final PrintWriter writer, final CategoricalData cate) {
    writer.write("@ATTRIBUTE " + cate.getCategoryName().replaceAll("\\s+", "-") + " {");
    for (int i = 0; i < cate.getNumOfCategories(); i++) {
      if (i != 0) {
        writer.write(",");
      }
      writer.write(addQuotes(cate.getOptionName(i)));
    }
    writer.write("}\n");
  }
}
