$(function() {
// ================ JS BELOW HERE ================
function getLocalBooleanValue(varname){
  var stringval = localStorage.getItem(varname) || 'false';
  var val = false;
  if(stringval === 'true'){
    val = true
  }
  return (val);
}
// ----
function shout(thing){
  //alert(thing);
}
// ----
function set_width(obj,ww){
  obj.width(ww);
}
// ----
function js_update_menubar_css(ww1,ww2){
  //alert('js_update_menubar_css');
  //console.log(ww1);
  $('#leftblock').width(ww1);
  $('#leftblock #sidebg').width(ww1);
  $('#leftblock #sidemenu').width(ww1);
  $('#leftblock #sidespacer').width(ww1);
  $('#rightblock').width(ww2);
  $('#rightblock').css('width','*');
  $('#rightblock').css('left',ww1);

  $('#leftblock').width(ww1);
  $('#leftblock #sidebg').width('100%');
  $('#leftblock #sidemenu').width('100%');
  $('#leftblock #sidespacer').width('100%');
  $('#rightblock').width(ww2);
  $('#rightblock').css('width','*');
  $('#rightblock').css('left',ww1);
  $('#rightblock #mainheader').width('100%');
  $('#rightblock #mainbody').width('100%');
  $('#rightblock #mainfooter').width('100%');
}
// ----
function js_animate_width(obj,dtime,nn,w1,w2) {
  var dt = dtime / nn;
  var dw = (w2-w1) / nn;
  var ww;
  for (let i = 0; i < nn; i++) {
    ww = w1 + dw*i;
    setTimeout(set_width(obj,ww),dt);
  }
}
// ----
function sleep(milliseconds) {
  const date = Date.now();
  let currentDate = null;
  do {
    currentDate = Date.now();
  } while (currentDate - date < milliseconds);
}
// ----
function js_animate_menubar(w1i,w1f,w2i,w2f,dt,ni,nn) {

    var ww1 = w1i + ni*(w1f - w1i)/nn;
    var ww2 = w2i + ni*(w2f - w2i)/nn;
    if(ni === 0){
      ww1 = w1i;
      ww2 = w2i;
    }
    if(ni === nn){
      ww1 = w1f;
      ww2 = w2f;
    }
    js_update_menubar_css(ww1,ww2);
    ++ni;
    if(ni <= nn){
      setTimeout(function(){
        //console.log(ni);
        js_animate_menubar(w1i,w1f,w2i,w2f,dt,ni,nn);
      }, dt);
    } else {
      if(w1f < w1i){
        sidebar_min();
      } else {
        sidebar_max();
        $('#leftblock #sidemenu .nav_label').css('display','inline-block');
      }
    }
}
// ----
function save_state() {
  if($("[name='sidebar_toggler']")[0].checked === false){
      $('#id_menubar_form')[0].elements["id_menubar_status"].value = 'unlocked';
      $obj_mbar_status = 'unlocked';
  } else {
    $('#id_menubar_form')[0].elements["id_menubar_status"].value = 'locked'
    $obj_mbar_status = 'locked';
  }
  if($('#blockcontainer').hasClass('sidebar_min')){
    $('#id_menubar_form')[0].elements["id_menubar_type"].value = 'sidebar_min';
    $obj_mbar_type = 'sidebar_min';
  }
  if($('#blockcontainer').hasClass('sidebar_max')){
    $('#id_menubar_form')[0].elements["id_menubar_type"].value = 'sidebar_max';
    $obj_mbar_type = 'sidebar_max';
  }
  if($('#blockcontainer').hasClass('sidebar_off')){
    $('#id_menubar_form')[0].elements["id_menubar_type"].value = 'sidebar_off';
    $obj_mbar_type = 'sidebar_off';
  }
}
// ----
function load_state() {
  $obj_mbar_type   = $('#id_menubar_form')[0].elements["id_menubar_type"].value;
  $obj_mbar_status = $('#id_menubar_form')[0].elements["id_menubar_status"].value;
  $obj_page_action = $('#id_menubar_form')[0].elements["id_page_action"].value;
}
// ----
function sidebar_min(){
  $('#id_menubar_form')[0].elements["id_menubar_type"].value = 'sidebar_min';
  $obj_mbar_type = 'sidebar_min';

  $('#blockcontainer').removeClass('sidebar_off');
  $('#blockcontainer').removeClass('sidebar_min');
  $('#blockcontainer').removeClass('sidebar_max');
  $('#blockcontainer').addClass('sidebar_min');
  $('#joinus_right').removeClass('display_none');
  $('#joinus_left').removeClass('display_none');
  $('#joinus_left').addClass('display_none');
}
function sidebar_max(){
  $('#id_menubar_form')[0].elements["id_menubar_type"].value = 'sidebar_max';
  $obj_mbar_type = 'sidebar_max';

  $('#blockcontainer').removeClass('sidebar_off');
  $('#blockcontainer').removeClass('sidebar_min');
  $('#blockcontainer').removeClass('sidebar_max');
  $('#blockcontainer').addClass('sidebar_max');
  $('#joinus_right').removeClass('display_none');
  $('#joinus_left').removeClass('display_none');
  $('#joinus_left').addClass('display_none');
}
function sidebar_off(){
  $('#id_menubar_form')[0].elements["id_menubar_type"].value = 'sidebar_off';
  $obj_mbar_type = 'sidebar_off';

  $('#leftblock').width(0);
  $('#leftblock #sidebg').width(0);
  $('#leftblock #sidemenu').width(0);
  $('#leftblock #sidespacer').width(0);
  $('#rightblock').css('width','100%');
  $('#rightblock').css('left',0);

  $('#blockcontainer').removeClass('sidebar_min');
  $('#blockcontainer').removeClass('sidebar_max');
  $('#blockcontainer').removeClass('sidebar_off');
  $('#blockcontainer').addClass('sidebar_off');
  $('#joinus_right').removeClass('display_none');
  $('#joinus_left').removeClass('display_none');
  $('#joinus_right').addClass('display_none');
}
// ----
function menu_on(anim=0) {
  if(anim > 0){
    menu_anim(mode='expand');
  }
  sidebar_max();
}
// ----
function menu_off(anim=0) {
  if(anim > 0){
    $('#leftblock #sidemenu .nav_label').css('display','none');
    menu_anim(mode='collapse');
  }
  sidebar_min();
}
// ----
function menu_anim(mode='expand') {
  var do_expand   = 0;
  var do_collapse = 0;
  var nn = anim_nstep;
  var dt = anim_time/nn;
  var w1i;
  var w2i;
  var w1f;
  var w2f;

  if(mode==='expand'){
    w1i = sidebar_width_off;
    w1f = sidebar_width_on;
    if($('#leftblock').width() === sidebar_width_off){
      do_expand = 1;
    }
  } else {
    w1i = sidebar_width_on;
    w1f = sidebar_width_off;
    if($('#leftblock').width() === sidebar_width_on){
      do_collapse = 1;
    }
  }

  w2i = window.innerWidth - w1i;
  w2f = window.innerWidth - w1f;

  if(do_expand === 1){
    js_animate_menubar(w1i,w1f,w2i,w2f,dt,ni=0,nn);
  }

  if(do_collapse === 1){
    js_animate_menubar(w1i,w1f,w2i,w2f,dt,ni=0,nn);
  }
}
// =================================
function hover_in_menu(){
  if($obj_mbar_status === 'unlocked'){
    menu_on(anim=1);
  }
}
// =================================
function hover_out_menu(){
  if($obj_mbar_status === 'unlocked'){
    menu_off(anim=1);
  }
}
// =================================
/*
setTimeout(()=> {
} ,200);
*/
// =================================
// How to create a bespoke function in Jquery:
jQuery.fn.extend({

  // check checkboxes
  my_new_function: function() {
    return this.each(function() {
      this.checked = true;
    });
  },
  enable_nav_clicks: function() {
    return this.each(function() {
      this.click(function (e) {
        save_state();
        $('#id_menubar_form')[0].action = $(this).attr('nav_action');
        $('#id_menubar_form')[0].submit();
      });
    });
  }

});
// Goes like:
// $(selector).my_new_function();
// =================================
function setup_clicks(){
  $('.has_modal_dropdown').click(function (e) {
    var eName  = $(event.target).attr('name');
    var eClass = $(event.target).attr('class');
    var eId    = $(event.target).attr('id');
    save_state();
    $obj_mbar_status = 'locked';
    var clicked_li_id = $(e.target).closest("li").attr('name');
    var elm = $(this);
    var xPos = e.pageX - elm.offset().left;
    var yPos = e.pageY - elm.offset().top;
    var new_modal_id = clicked_li_id + '_modal';
    var new_modal_parent_id = '#modal_carpark';
    var new_modal_selector = new_modal_parent_id + ' #' + new_modal_id;
    // check if modal already exists:
    if($(new_modal_selector).length === 0){
      var new_modal = $('#empty_modal_template');
      new_modal.attr('id',new_modal_id);
      $(new_modal_parent_id).append(new_modal);
      new_modal = $(new_modal_selector);
      //alert(new_modal_selector);
      // Get the content for the modal:
      var new_content = $("[name='" + clicked_li_id + "'] .dropdown");
      $(new_modal_selector + ' span.modal-content').html(new_content.html());
    } else {
      new_modal = $(new_modal_selector);
    }
    //new_modal[0].style.display = "flex";

    new_modal.css('display','inline-block');
    var modal_content = $(new_modal_selector + ' span.modal-content');
    modal_content.hover(
      function() {
        // do nothing on hover-in
      }, function() {
        new_modal.css('display','none');
        load_state();
      }
    );
    if(window.innerWidth < win_size_thresh){
      modal_content.css('left',e.pageX - modal_content.width());
    } else {
      modal_content.css('left',e.pageX);
    }
    modal_content.css('top',e.pageY);
    modal_content.find('.nav_clicks').click(function (e) {
      save_state();
      $('#id_menubar_form')[0].action = $(this).attr('nav_action');
      $('#id_menubar_form')[0].submit();
    });
  });
  // -----
  // Order is important.
  // The above has to exist before you can add a click listener to it.
  $("[name='nav_menu']").find('.nav_clicks').click(function (e) {
    save_state();
    $('#id_menubar_form')[0].action = $(this).attr('nav_action');
    $('#id_menubar_form')[0].submit();
  });
  //$("[name='nav_menu']").find('.nav_clicks').enable_nav_clicks();



}
// -----
function show_setting(sel,varname){
  var str;
  var val;
  var css = getComputedStyle($(sel)[0]);
  for (var i = 0; i < css.length; i++) {
    if(css[i] === varname){
      str = css[i];
      val = css.getPropertyValue(css[i]);
    }
  }
  str = sel + ' : ' + str + ' = ' + val + '\n';
  return (str);
}
// -----
function show_settings(){
  var str = 'Settings: \n';
  str = str + show_setting('#rightblock','width');
  str = str + show_setting('#mainheader','width');
  str = str + show_setting('#mainbody'  ,'width');
  str = str + show_setting('#mainfooter','width');
  alert(str);
}
// =================================
function setup_side_menu(){
  if($obj_mbar_type === 'sidebar_min'){
    sidebar_min();
  } else {
    sidebar_max();
  }
  $('.nav_hover').hover(
    function() {
      hover_in_menu();
    }, function() {
      // do nothing on hover-out
    }
  );
  $('#sidemenu').hover(
    function() {
      // do nothing on hover-in
    }, function() {
      hover_out_menu();
    }
  );
  setup_clicks();
  js_update_menubar_css(ww1,ww2);
}
// =====
function setup_top_menu(){
  sidebar_off();
  $('.nav_hover').hover(
    function() {
      // do nothing on hover-in
    }, function() {
      // do nothing on hover-out
    }
  );
  setup_clicks();
}
// =================================
function getCSS(selector) {
    var css_data = {};
    var css_obj = getComputedStyle($(selector)[0]);
    for (var i = 0; i < css_obj.length; i++) {
        css_data[css_obj[i]] = css_obj.getPropertyValue(css_obj[i]);
    }
    return css_data;
}
// =================================
function compareObjs(obj1,obj2){
  var new_obj = {};
  Object.keys(obj1).forEach(key => {
    if(obj1[key] != obj2[key]){
      new_obj[key] = '[' + obj1[key] + '] [' + obj2[key] + ']'
    }
  });
  return (new_obj);
}
// =================================
// =================================
//var sidebar_status = getLocalBooleanValue('sidebar_status');
//var sidebar_lock   = getLocalBooleanValue('sidebar_lock');
var $obj_mbar_type;
var $obj_mbar_status;
var $obj_page_action;
load_state();
var win_size_thresh = 600;
var state_change = 0;
var this_val = 1;
var prev_val = 1;
var anim_time = 10;
var anim_nstep = 50;
var ww1;
var ww2;
var anim_dt = anim_time / anim_nstep;
var lblock = $('#leftblock');
var rblock = $('#rightblock');
var gcs = getComputedStyle(document.documentElement);
var sidebar_width_off = gcs.getPropertyValue('--sidebar_width_off');
var sidebar_width_on  = gcs.getPropertyValue('--sidebar_width_on');
sidebar_width_off = Number(sidebar_width_off.replace('px',''));
sidebar_width_on  = Number(sidebar_width_on.replace('px',''));
// =================================
// Set initial layout:

if(window.innerWidth < win_size_thresh){
  setup_top_menu();
} else {
  setup_side_menu();
}
// Ensure consistency between form options and the layout
if($obj_mbar_type === 'sidebar_max'){
  if($('#menubar').hasClass('sidebar_min')) menu_on(anim=0);
} else {
  if($('#menubar').hasClass('sidebar_max')) menu_off(anim=0);
}

//$('#infobox').html('done <br/>' + JSON.stringify(new_obj));
// =================================
$("[name='sidebar_toggler']").click(function () {
  if($(this)[0].checked === true){
    $("[name='nav_label_lock']").removeClass('nav_label_unlocked').addClass('nav_label_locked');
    $obj_mbar_status = 'locked';
  } else {
    $("[name='nav_label_lock']").removeClass('nav_label_locked').addClass('nav_label_unlocked');
    $obj_mbar_status = 'unlocked';
  }
});
// =================================
// =================================



// =================================
// When the user clicks anywhere outside of the modal, close it
window.onclick = function(event) {
  var eName  = $(event.target).attr('name');
  var eClass = $(event.target).attr('class');
  var eId    = $(event.target).attr('id');
  //alert(eName + ' : ' + eClass + ' : ' + eId);
  if($(event.target).attr('click_action') === 'dismiss_modal') {
    $(event.target).css('display','none');
  }
}
// =================================
// check for a state-change before changing the layout;
function adjustForResize(){
  var winX = window.innerWidth;
  this_val = winX - win_size_thresh;
  if(this_val <= 0){
    this_val = -1;
  } else {
    this_val = 1;
  }

  if(prev_val != this_val){
    state_change = 1;
  }
  if(state_change === 1){
    //show_settings();
      if(winX <= win_size_thresh){

        $('#leftblock').width(0);
        $('#leftblock #sidebg').width(0);
        $('#leftblock #sidemenu').width(0);
        $('#leftblock #sidespacer').width(0);
        $('#rightblock').width(winX);
        $('#rightblock').css('width','100%');
        $('#rightblock').css('left',0);

        setup_top_menu();
        $('#id_menubar_form')[0].elements["id_menubar_type"].value = 'sidebar_off';
        $obj_mbar_type = 'sidebar_off';
        $('#id_menubar_form')[0].submit();
        this_val = -1;
        prev_val = -1;
        state_change = 0;
      } else {
        $('#leftblock').width(sidebar_width_on);
        $('#leftblock #sidebg').width(sidebar_width_on);
        $('#leftblock #sidemenu').width(sidebar_width_on);
        $('#leftblock #sidespacer').width(sidebar_width_on);
        $('#rightblock').width(winX - sidebar_width_on);
        $('#rightblock').css('width','*');
        $('#rightblock').css('left',sidebar_width_on);
        setup_side_menu();

        $('#id_menubar_form')[0].elements["id_menubar_type"].value = 'sidebar_max';
        $obj_mbar_type = 'sidebar_max';
        $('#id_menubar_form')[0].submit();
        this_val = 1;
        prev_val = 1;
        state_change = 0;
      }
  }
  prev_val = this_val;
}
window.addEventListener("resize", adjustForResize);
// ================ JS ABOVE HERE ================
});
