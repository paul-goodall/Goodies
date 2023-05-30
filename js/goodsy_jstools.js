// ===============
function cleanText(arr) {
    var arr2 = [], i;
    for(i = 0; i < arr.length; i++)
        if (arr[i] != '')
            arr2.push(arr[i]);
    return arr2;
}
// ===============
function getAllIndexes(arr, val) {
    var indexes = [], i;
    for(i = 0; i < arr.length; i++)
        if (arr[i] === val)
            indexes.push(i);
    return indexes;
}
// ===============
function jsWhich(arr, val) {
    var indexes = [], i;
    for(i = 0; i < arr.length; i++)
        if (arr[i] === val){
          indexes.push(1);
        } else {
          indexes.push(0);
        }
    return indexes;
}
// ===============
function jsWhere(arr, ii) {
    var arr2 = [], i;
    for(i = 0; i < ii.length; i++){
      arr2.push(arr[ii[i]]);
    }
    return arr2;
}
// ===============
function ArraySlice(v1,i1,i2) {
    var arr2 = [];
    var imin = Math.max(i1,0);
    var imax = Math.min(i2,(v1.length-1));
    for(let i = imin; i <= imax; i++){
      arr2.push(v1[i]);
    }
    return arr2;
}
// ===============
// ===============
function SumArray(v1) {
    var v = 0;
    for(let i = 0; i < v1.length; i++){
      v += v1[i];
    }
    return v;
}
// ===============
function ArrayMultiply(v1, v2) {
    var v3 = [], i;
    for(i = 0; i < v1.length; i++){
      v3[i] = v1[i]*v2[i];
    }
    return v3;
}
// ===============
function ArrayDivide(v1, v2) {
    var v3 = [], i;
    for(i = 0; i < v1.length; i++){
      v3[i] = v1[i]/v2[i];
    }
    return v3;
}
// ===============
function getKeys(obj) {
  var my_keys = [];
  Object.keys(obj).forEach(key => {
  my_keys.push(key);
  });
  return my_keys;
}
// ===============
// ===============================
function alert_obj(obj){
  var str = '';
  Object.keys(obj).forEach(key => {
    str = str + key + ' : ' + obj[key] + '\n ';
  });
  alert(str);
}
// ===============
function obj2string(v1) {
  var my_str = '';
  for(let i = 0; i < v1.length; i++){
    my_str = my_str + v1[i] + ','
  }
  my_str = my_str.substring(0,my_str.length-1);
  return my_str;
}
// ===============
// ============================================
// HTML table to DF:
// -----
// provide the tag to a html table and it returns a javascript object like a 
// data table with columns and rows.
function html_tbl_to_df(data_tbl_tag) {
  let data_tbl = $(data_tbl_tag);
  let nr = $(data_tbl_tag + ' tr').length;
  let nc = $(data_tbl_tag + ' tr th').length;
  // first row is the header:
  let data = [...data_tbl[0].rows].map(t => [...t.children].map(u => u.innerText));
  data = obj2string(data);
  data = data.split(',');
  df = {};
  for(j = 0; j < nr; j++){
    for(i = 0; i < nc; i++){
      let z = j*nc + i;
      if(j == 0){
        df[data[i]] = [];
      } else {
        df[data[i]].push(data[z]);
      }
    }
  }
  return (df);
}
// ============================================