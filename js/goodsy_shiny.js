// ============================================
// Globals: things that don't depend on anything else:
// -----
var id_img_frame = 'frame_image';
var id_img_zoom  = 'frame_zoombox';
var id_frame_list = 'id_frame_list';
var frame_nx = 1920;
var frame_ny = 1080;
var frame_zoomfac = 1.1;
var im_nx = Math.round(frame_zoomfac * frame_nx);
var im_ny = Math.round(frame_zoomfac * frame_ny);
var actual_zoomfac_x = im_nx/frame_nx;
var actual_zoomfac_y = im_ny/frame_ny;
var var_plot_objs;
var draw_markers = 1;

var counter_frame = 0;
var num_frames = 2;
var frame_list = [];
// ============================================
// This comes first:
$(document).ready(function() {
 //alert('document ready');
});
// ============================================
// Do something as soon as shiny is connected:
// -----
$(document).on('shiny:connected', function(event) {
  //alert('Connected to the server');
  
  // Can only call to these once the objects themselves have loaded:
  var img = document.getElementById(id_img_frame);
  var img_jq = $('#'+id_img_frame);
  var result = document.getElementById(id_img_zoom);

  img_jq.css('width',  im_nx);
  img_jq.css('height', im_ny);
  img_jq.css('margin',  0);
  img_jq.css('padding', 0);
  img_jq.css('border',  0);
  imageZoom(id_img_frame, id_img_zoom);
  
});
// ============================================
  // ============================================
// Return the keypress back from JS to shiny as the variable: input$keynum[1]
// Note that input$keynum[2] returns a random float, and ensures that reactivity
// continues even when the same key is pressed successively.
// -----
$(document).on('keypress', function (e) {
  let img = document.getElementById(id_img_frame);
  let result = document.getElementById(id_img_zoom);
  let str_frame_list = document.getElementById(id_frame_list).innerHTML;
  $('#datapoints_all table').css('height', 0);
  $('#datapoints_all table tr').css('height', 0);
  frame_list = str_frame_list.split(',');
  num_frames = frame_list.length;
  let key_num  = e.which;
  let key_code = e.code;
  let key_char = e.key;
  Shiny.onInputChange('key_num',  [key_num, Math.random() ]);
  Shiny.onInputChange('key_code', [key_code, Math.random()]);
  Shiny.onInputChange('key_char', [key_char, Math.random()]);
  //alert(key_char);
  if (key_char === 'w') {
    ++counter_frame;
    counter_frame = Math.min(counter_frame, num_frames-1);
  }
  if (key_char === 'q') {
    --counter_frame;
    counter_frame = Math.max(counter_frame, 0);
  }
  if (key_char === 'x') {
    // toggle markers:
    draw_markers *= -1;
  }
  let this_frame = frame_list[counter_frame];
  let frame_label = 'frame_' + String(counter_frame).padStart(5, '0');
  Shiny.onInputChange('var_frame_num', [counter_frame, Math.random()]);
  Shiny.onInputChange('var_frame_src', [this_frame, Math.random()]);
  document.getElementById('frame_name').innerHTML = this_frame;
  img.src = this_frame;
  result.style.backgroundImage = "url('" + this_frame + "')";
  
  //clear_reset_div_to_img_only();
  clear_plot_objects();
  if(draw_markers == 1){
    plot_frame_boxes(counter_frame);
    if(counter_frame > 0){
      plot_frame_boxes(counter_frame-1, box_dx=20, box_dy=20, colour='yellow');
    }
  }
  
});
// ============================================
// Comments:
// -----
$(document).on('click', function (e) {
  let img = document.getElementById(id_img_frame);
  let cp = global_getCursorPos(img,e);
  let img_x = cp.x;
  let img_y = im_ny - cp.y;
  let frame_x = img_x/actual_zoomfac_x;
  let frame_y = img_y/actual_zoomfac_y;
  let var_frame_inside = 1;
  if (frame_x < 0){
    frame_x = 0;
    var_frame_inside = 0;
  }
  if (frame_x > frame_nx){
    frame_x = frame_nx;
    var_frame_inside = 0;
  }
  if (frame_y < 0){
    frame_y = 0;
    var_frame_inside = 0;
  }
  if (frame_y > frame_ny){
    frame_y = frame_ny;
    var_frame_inside = 0;
  }
  if (var_frame_inside == 1){
    plot_box('.img-zoom-container', 'test', box_x=img_x, box_y=img_y, box_dx=10, box_dy=10,colour='cyan');
  }
  Shiny.onInputChange('var_frame_inside', [var_frame_inside, Math.random()]);
  Shiny.onInputChange('var_frame_x', [frame_x, Math.random()]);
  Shiny.onInputChange('var_frame_y', [frame_y, Math.random()]);
  document.getElementById('frame_x').innerHTML = frame_x;
  document.getElementById('frame_y').innerHTML = frame_y;
});
// ============================================
// Plot a box:
// -----
function plot_box(parent_tag, box_id, box_x, box_y, box_dx=10, box_dy=10,colour='red') {
  let tmp = $('<div class="box"></div>');
  let parent = $(parent_tag);
  let pbox = parent[0].getBoundingClientRect();
  //alert(Object.entries(pbox.toJSON()));
  tmp.attr('id', box_id);
  tmp.attr('name', 'plot_object');
  // set some sizes:
  let border = 1;
  let padding = 0;
  let margin = 0;
  let box_total_w = box_dx + 2*(border+padding+margin);
  let box_total_h = box_dy + 2*(border+padding+margin);
  tmp.css('width',  box_dx);
  tmp.css('height', box_dy);
  tmp.css('border-width', border);
  tmp.css('border-color', colour);
  tmp.css('margin', margin);
  tmp.css('padding', padding);
  let x = box_x + pbox.left - 0.5*box_total_w + 1;
  let y = pbox.bottom - box_y - 0.5*box_total_h + 1;
  // 1256 : 223.64 : 22 : 1022.3600000000001
  //alert(pbox.bottom + ' : ' + box_y + ' : ' + box_total_h + ' : ' + y);
  tmp.css('left', x);
  tmp.css('top' , y);
  parent.append(tmp);
}
// Plot a box:
// -----
function plot_square(parent_tag, box_id, box_x, box_y, box_dx=10, box_dy=10,colour='red') {
  let tmp = $('<div class="square"></div>');
  let parent = $(parent_tag);
  let pbox = parent[0].getBoundingClientRect();
  //alert(Object.entries(pbox.toJSON()));
  tmp.attr('id', box_id);
  tmp.attr('name', 'plot_object');
  // set some sizes:
  let border = 0;
  let padding = 0;
  let margin = 0;
  let box_total_w = box_dx + 2*(border+padding+margin);
  let box_total_h = box_dy + 2*(border+padding+margin);
  tmp.css('width',  box_dx);
  tmp.css('height', box_dy);
  tmp.css('border-width', border);
  tmp.css('border-color', colour);
  tmp.css('margin', margin);
  tmp.css('padding', padding);
  let x = box_x + pbox.left - 0.5*box_total_w + 1;
  let y = box_y + pbox.top  - 0.5*box_total_h + 1;
  tmp.css('left', x);
  tmp.css('top' , y);
  parent.append(tmp);
}
// ============================================
// Clear the parent div container, except for the img:
// -----
function clear_reset_div_to_img_only(){
  let img = document.getElementById(id_img_frame);
  $('.img-zoom-container').empty();
  $('.img-zoom-container').append(img);
}
// ============================================
// Remove all box and square objects:
// -----
function clear_objects(tag){
  $(tag).each(function(i, obj) {
    obj.remove();
  });
}
// -----
function clear_plot_objects(){
  let tag = '.img-zoom-container div[name="plot_object"]';
  clear_objects(tag);
}
// -----
function remove_objects(tag){
  let objs = $(tag);
  $(tag).each(function(i, obj) {
    obj.remove();
  });
  return (objs);
}
// -----
function remove_plot_objects(){
  let tag = '.img-zoom-container div[name="plot_object"]';
  let plot_objs = remove_objects(tag);
  return (plot_objs);
}
// -----
function append_plot_objects(plot_objs){
  let parent = $('.img-zoom-container');
  plot_objs.each(function(i, obj) {
    parent.append(obj);
  });
}
// -----
// ============================================
// Plot all the boxes:
// -----
function plot_frame_boxes(counter_frame, box_dx=10, box_dy=10, colour='red') {
  let frame_label = 'frame_' + String(counter_frame).padStart(5, '0');
  let data = html_tbl_to_df('#datapoints_all table');
  //alert_obj(data);
  let ii = getAllIndexes(data['frame'], frame_label);
  //alert(SumArray(ii));
  if(SumArray(ii) > 0){
    let xx = jsWhere(data['x'], ii);
    let yy = jsWhere(data['y'], ii);
    //alert(xx);
    for(let i = 0; i < xx.length; i++){
      let my_box_id = frame_label + '_box_' + i; 
      let my_x = xx[i]*1*actual_zoomfac_x;
      let my_y = yy[i]*1*actual_zoomfac_y;
      //alert(my_box_id + ' : ' + my_x + ' : ' + my_y);
      plot_box('.img-zoom-container', my_box_id, my_x, my_y, box_dx=box_dx, box_dy=box_dy, colour=colour);
    }
  }
}
// ============================================
// Do something as soon as shiny is connected:
// -----
function global_getCursorPos(img,e) {
    var a, x = 0, y = 0;
    e = e || window.event;
    /*get the x and y positions of the image:*/
    a = img.getBoundingClientRect();
    /*calculate the cursor's x and y coordinates, relative to the image:*/
    x = e.pageX - a.left;
    y = e.pageY - a.top;
    /*consider any page scrolling:*/
    x = x - window.pageXOffset;
    y = y - window.pageYOffset;
    return {x : x, y : y};
}
// ============================================
// Comments:
// -----
function imageZoom(imgID, resultID) {
  var img, lens, result, cx, cy;
  img = document.getElementById(imgID);
  result = document.getElementById(resultID);
  /*create lens:*/
  lens = document.createElement("DIV");
  lens.setAttribute("class", "img-zoom-lens");
  /*insert lens:*/
  img.parentElement.insertBefore(lens, img);
  /*calculate the ratio between result DIV and lens:*/
  cx = result.offsetWidth / lens.offsetWidth;
  cy = result.offsetHeight / lens.offsetHeight;
  /*set background properties for the result DIV:*/
  result.style.backgroundImage = "url('" + img.src + "')";
  result.style.backgroundSize = (img.width * cx) + "px " + (img.height * cy) + "px";
  /*execute a function when someone moves the cursor over the image, or the lens:*/
  lens.addEventListener("mousemove", moveLens);
  img.addEventListener("mousemove", moveLens);
  /*and also for touch screens:*/
  lens.addEventListener("touchmove", moveLens);
  img.addEventListener("touchmove", moveLens);
  function moveLens(e) {
    var pos, x, y;
    /*prevent any other actions that may occur when moving over the image:*/
    e.preventDefault();
    /*get the cursor's x and y positions:*/
    pos = getCursorPos(e);
    /*calculate the position of the lens:*/
    x = pos.x - (lens.offsetWidth / 2);
    y = pos.y - (lens.offsetHeight / 2);
    /*prevent the lens from being positioned outside the image:*/
    if (x > img.width - lens.offsetWidth) {x = img.width - lens.offsetWidth;}
    if (x < 0) {x = 0;}
    if (y > img.height - lens.offsetHeight) {y = img.height - lens.offsetHeight;}
    if (y < 0) {y = 0;}
    /*set the position of the lens:*/
    lens.style.left = x + "px";
    lens.style.top = y + "px";
    /*display what the lens "sees":*/
    result.style.backgroundPosition = "-" + (x * cx) + "px -" + (y * cy) + "px";
  }
  function getCursorPos(e) {
    var a, x = 0, y = 0;
    e = e || window.event;
    /*get the x and y positions of the image:*/
    a = img.getBoundingClientRect();
    /*calculate the cursor's x and y coordinates, relative to the image:*/
    x = e.pageX - a.left;
    y = e.pageY - a.top;
    /*consider any page scrolling:*/
    x = x - window.pageXOffset;
    y = y - window.pageYOffset;
    return {x : x, y : y};
  }
}
// ============================================
// Comments:
// -----
// ============================================
// Comments:
// -----
// ============================================
// Comments:
// -----
// ============================================
// Comments:
// -----
// ============================================
// Comments:
// -----
// ============================================
// Comments:
// -----
// ============================================
// Comments:
// -----
// ============================================
// Comments:
// -----
// ============================================
// Comments:
// -----
// ============================================
// Comments:
// -----
// ============================================
// Comments:
// -----